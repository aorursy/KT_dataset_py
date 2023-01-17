import streamlit as st

#NLP Packages
import spacy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import gensim
from gensim import summarization
from gtts import gTTS
from IPython.display import Audio

vid=open("advertisement.mp4","rb")
st.video(vid)
st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)

def text_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)
    #tokens = [token.text for token in docx]
    allData = [('"tokens":{},\n"Lemma":{}'.format(token.text,token.lemma_)) for token in docx]
    return allData

def entity_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_) for entity in docx.ents ]
    allData = ['"Tokens":{},\n"Entities":{}'.format(tokens, entities)]
    return allData

def hindi_trans(docx):
    tb = TextBlob(docx)
    result =tb.translate(to='hi')
    return result


def french_trans(docx):
    tb = TextBlob(docx)
    result =tb.translate(to='fr')
    return result

def telugu_trans(docx):
    tb = TextBlob(docx)
    result =tb.translate(to='te')
    return result

def tamil_trans(docx):
    tb = TextBlob(docx)
    result =tb.translate(to='ta')
    return result

#sumy packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def sumy_lexrank_summarizer(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    lex_summary = lex_summarizer(parser.document, 3)
    lex_summary_list = [str(sentence) for sentence in lex_summary]
    result = ' '.join(lex_summary_list)
    return result

def sumy_luhn_summarizer(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    luhn_summarizer = LuhnSummarizer()
    luhn_summary = luhn_summarizer(parser.document, 3)
    luhn_summary_list = [str(sentence) for sentence in luhn_summary]
    result = ' '.join(luhn_summary_list)
    return result

def sumy_lsa_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lsa_summarizer = LsaSummarizer()
    lsa_summary = lsa_summarizer(parser.document, 3)
    lsa_summary_list = [str(sentence) for sentence in lsa_summary]
    result = ' '.join(lsa_summary_list)
    return result

def alt_sumy_using_spacy(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    summarizer_lsa2 = LsaSummarizer()
    summarizer_lsa2 = LsaSummarizer(Stemmer("english"))
    summarizer_lsa2.stop_words = get_stop_words("english")
    alt_summary = summarizer_lsa2(parser.document, 3)
    alt_summary_list = [str(sentence) for sentence in alt_summary]
    result = ' '.join(alt_summary_list)
    return result

def main():
    #st.title("StanShare")
    html_temp = """
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:yellow;text-align:center;">NLP Task</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)


    st.sidebar.subheader("Designed By")
    st.sidebar.text("Karteek Menda")

    image = Image.open('QRCode.png')
    st.sidebar.image(image, use_column_width=True)
    st.sidebar.text("@Copyrights Reserved...")


    # Tokenization
    if st.checkbox("Show Token and Lemma"):
        st.subheader("Tokenize your text")
        message = st.text_area("Enter your text here", "Type Here")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # Named Entity Recognition
    if st.checkbox("Named Entities"):
        st.subheader("Extract entities from your text")
        message = st.text_area("Enter your text here", "Type Here")
        if st.button("Extract"):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)

    # Sentiment Analysis
    if st.checkbox("Sentiment Analysis"):
        st.subheader("Sentiment of your text")
        message = st.text_area("Enter your text here", "Type Here")
        if st.button("Analyze"):
            blob = TextBlob(message, analyzer=NaiveBayesAnalyzer())
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    # Language translation.
    if st.checkbox("Translate to your native Language"):
        st.subheader("Translation")
        message = st.text_area("Enter your text here", "Type Here")
        language_options = st.selectbox("choice of your language", ("Hindi", "French", "Telugu", "Tamil"))
        if st.button("Translate"):
            if language_options =="Hindi":
                st.text("To Hindi.......")
                result_translation  = hindi_trans(message)
            elif language_options == "French":
                st.text("To French.......")
                result_translation = french_trans(message)
            elif language_options == "Telugu":
                st.text("To Telugu......")
                result_translation = telugu_trans(message)
            elif language_options == "Tamil":
                st.text("To Tamil......")
                result_translation = tamil_trans(message)
            else:
                st.text("To Hindi.......")
                result_translation = hindi_trans(message)

            st.success(result_translation)

    #Text summarization
    if st.checkbox('Summary of your TEXT'):

        message = st.text_area("Enter your TEXT here", "Type Here")
        summary_options = st.selectbox("Choice of your summarizer", ("Gensim summarizer", "Sumy_lexrank_summarizer", "Sumy_luhn_summarizer", "Sumy_lsa_summarizer", "alt_sumy_using_stopwords"))
        if st.button("summarize"):
            if summary_options == 'Gensim summarizer':
                st.text("Using Gensim...")
                result_summary = gensim.summarization.summarize(message)

            elif summary_options == 'Sumy_lexrank_summarizer':
                st.text("Using sumy_lexrank_summarizer...")
                result_summary = sumy_lexrank_summarizer(message)

            elif summary_options == 'Sumy_luhn_summarizer':
                st.text("Using sumy_luhn_summarizer...")
                result_summary = sumy_luhn_summarizer(message)

            elif summary_options == 'Sumy_lsa_summarizer':
                st.text("Using sumy_lsa_summarizer...")
                result_summary = sumy_lsa_summarizer(message)

            elif summary_options == 'alt_sumy_using_spacy':
                st.text("Using alt_sumy_using_stopwords...")
                result_summary = alt_sumy_using_spacy(message)

            else:
                st.warning("Using Defaut Sumarizer")
                st.text("Using Gensim")
                result_summary = gensim.summarization.summarize(message)


            st.success(result_summary)

if __name__=='__main__':
    main()
