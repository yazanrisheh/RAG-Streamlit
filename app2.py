from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from PyPDF2 import PdfReader

load_dotenv()
raw_text = None

def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    document_chunks = text_splitter.split_text(text)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_texts(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0.1, model_name='gpt-3.5-turbo')

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(temperature=0.1, model_name='gpt-3.5-turbo')

    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']

# app config
st.set_page_config(page_title="RAAFYA AI", page_icon="🤖")
st.title("RAAFYA AI")
st.caption("Remember the answers of RAAFYA is not always accurate")

# sidebar
with st.sidebar:
    st.header("Settings")
    username = st.text_input("Username: ")
    password = st.text_input("Password: ", type="password")
    if username != "ali123" or password != "ali12345":
        st.write("Incorrect credentials")
        pdf_doc = st.file_uploader(label="Upload your PDFs", disabled=True)
    else:
        st.subheader("Your Documents")
        pdf_doc = st.file_uploader(label="Upload your PDFs",
                                    accept_multiple_files=False,
                                      help="nice",
                                        disabled=False,
                                          label_visibility="hidden")
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_doc)
                st.session_state.raw_text = raw_text

if pdf_doc is None:
    st.info("Please sign in and upload your docs")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am RAAFYA. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = raw_text

    # user input
    user_input = st.chat_input("Type your message here...")
    if user_input is not None and user_input != "":
        response = get_response(user_input)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)