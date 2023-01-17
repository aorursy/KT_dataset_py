!pip install rank_bm25 -q
!pip install pytextrank
!pip install summa
import nltk
nltk.download('punkt') # one time execution
nltk.download('stopwords')
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import networkx as nx
from rank_bm25 import BM25Okapi
import logging
import pytextrank
import sys
import spacy
from summa import keywords
from summa.summarizer import summarize

# function to rank documents using BM25Okapi
def rank_docs(df_col, query, n):
    corpus = df_col.values
    corpus_token = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(corpus_token)
    
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_doc = bm25.get_top_n(tokenized_query, corpus, n)
    
    return top_doc

stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

# function to clean the sentences
def clean_sentence(top_doc):
    sentences = []
    for s in top_doc:
        sentences.append(sent_tokenize(s))
    
    sentences = [y for x in sentences for y in x] # flatten the list
    
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    
    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    
    # remove stopwords from sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    
    return sentences, clean_sentences
    

# function to extract word vectors 
def extract_word_vectors():
    word_embeddings = {}
    os.chdir('/kaggle/working') # change directory
    f = open('glove.6B.100d.txt', encoding = 'utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    return word_embeddings

# function to find sentence vectors
def find_sentence_vec(clean_sentences):

    word_embeddings = extract_word_vectors()
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    
    return sentence_vectors

# function to extract summary of documents
def find_ranked_sent(df_col, query, n):
    top_doc = rank_docs(df_col, query, n)
    sentences, clean_sentences = clean_sentence(top_doc)
    sentence_vectors = find_sentence_vec(clean_sentences)
    # initialize similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
                
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse = True)
    
    return ranked_sentences
    
def summarize_docs(top_docs, num_words):
    text = ' '.join([str(x) for x in top_docs]) # join the docs together
    summary = summarize(text, words=num_words)
    return summary
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
os.chdir('/kaggle/input/CORD-19-research-challenge')
meta_df = pd.read_csv('metadata.csv', dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str,
    'full_text_file':str,
    'abstract':str,
    'title':str
})

df = meta_df[['abstract']]
df = df.drop_duplicates(['abstract'])
df = df.dropna()
df_col = df.abstract
query = "COVID-19 risk factors"
ranked_sentences = find_ranked_sent(df_col, query, n = 10)
risk_factors = rank_docs(df_col, query = 'COVID-19 risk factors', n=5)
risk_factors
summarize_docs(risk_factors, num_words=200)
epidemiology = rank_docs(df_col, query = 'COVID-19 epidemiological studies', n=5)
epidemiology
summarize_docs(epidemiology, num_words=300)
smoking = rank_docs(df_col, query = 'COVID-19 smoking pre-existing pulmonary disease', n=5)
smoking
summarize_docs(smoking, num_words=200)
coinfections = rank_docs(df_col, query = 'COVID-19 co-infections co-morbidities co-existing respiratory viral infections', n=5)
coinfections
summarize_docs(coinfections, num_words=200)
neonates = rank_docs(df_col, query = 'COVID-19 noenates pregnant women', n=5)
neonates
summarize_docs(neonates, num_words=200)
behavioral = rank_docs(df_col, query = 'COVID-19 socioeconomic behavioral factors to understand economic impact', n=5)
behavioral
summarize_docs(behavioral, num_words = 200)
transmission = rank_docs(df_col, query = 'COVID-19 transmission dynamics of the virus, including basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors', n=5)
transmission
summarize_docs(transmission, num_words=200)
severity = rank_docs(df_col, query = 'COVID-19 severity, risk of fatality among symptomatic hospitalized patients, and high-risk patient groups', n=5)
severity
summarize_docs(severity, num_words=200)
susceptible = rank_docs(df_col, query = 'COVID-19 susceptibility of populations', n=5)
susceptible
summarize_docs(susceptible, num_words=200)
mitigation = rank_docs(df_col, query = 'COVID-19 public health mitigation measures that could be effective for control', n=5)
mitigation
summarize_docs(mitigation, num_words=300)