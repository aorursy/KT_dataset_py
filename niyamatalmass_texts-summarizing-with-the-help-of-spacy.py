import numpy as np # linear algebra
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display
import base64
import string
import re
from collections import Counter
from time import time
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from nltk.corpus import stopwords
import nltk
import heapq
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("lda").setLevel(logging.WARNING)

stopwords = stopwords.words('english')
sns.set_context('notebook')
df_questions = pd.read_csv("../input/Questions.csv", nrows=5000,usecols =['Score', 'Title', 'Body'],encoding='latin1')
df_questions = df_questions.dropna()
df_questions.head(15)
!python -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')
def normalize_text(text):
    tm1 = re.sub('<pre>.*?</pre>', '', text, flags=re.DOTALL)
    tm2 = re.sub('<code>.*?</code>', '', tm1, flags=re.DOTALL)
    tm3 = re.sub('<[^>]+>', '', tm1, flags=re.DOTALL)
    return tm3.replace("\n", "")
# in this step we are going to remove code syntax from text 
df_questions['Body_Cleaned_1'] = df_questions['Body'].apply(normalize_text)
print('Before normalizing text::::::::::\n')
print(df_questions['Body'][2])
print('\nAfter normalizing text:::::::::\n')
print(df_questions['Body_Cleaned_1'][2])
# Clean text before feeding it to spaCy
punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs, logging=False):
    texts = []
    doc = nlp(docs, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    tokens = ' '.join(tokens)
    texts.append(tokens)
    return pd.Series(texts)


df_questions['Body_Cleaned'] = df_questions['Body_Cleaned_1'].apply(lambda x: cleanup_text(x, False))
print('Question Body with punctuatin and stopwords:::::::::::\n')
print(df_questions['Body_Cleaned_1'][0])
print('\nQuestion Body after removing punctuation and stopwrods:::::::::::::\n')
print(df_questions['Body_Cleaned'][0])
plt.subplot(1, 2, 1)
(df_questions['Score']).plot.hist(bins=30, figsize=(30,5), edgecolor='white',range=[0,250])
plt.xlabel('Number of upvotes', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Number of upvotes distribution', fontsize=17)
plt.show()
df_questions['Title_len'] = df_questions['Body_Cleaned'].str.split().str.len()
df = df_questions.groupby('Title_len')['Score'].mean().reset_index()
trace1 = go.Scatter(
    x = df['Title_len'],
    y = df['Score'],
    mode = 'lines+markers',
    name = 'lines+markers'
)
layout = dict(title= 'Average Upvote by Question Body Length',
              yaxis = dict(title='Average Upvote'),
              xaxis = dict(title='Question Body Length'))
fig=dict(data=[trace1], layout=layout)
py.iplot(fig)

# this is function for text summarization

def generate_summary(text_without_removing_dot, cleaned_text):
    sample_text = text_without_removing_dot
    doc = nlp(sample_text)
    sentence_list=[]
    for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
        sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}  
    for word in nltk.word_tokenize(cleaned_text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]


    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print("Original Text::::::::::::\n")
    print(text_without_removing_dot)
    print('\n\nSummarized text::::::::\n')
    print(summary)  
generate_summary(df_questions['Body_Cleaned_1'][3], df_questions['Body_Cleaned'][3])
generate_summary(df_questions['Body_Cleaned_1'][5], df_questions['Body_Cleaned'][5])
generate_summary(df_questions['Body_Cleaned_1'][67], df_questions['Body_Cleaned'][68])