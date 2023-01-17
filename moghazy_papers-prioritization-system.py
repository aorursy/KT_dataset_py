!pip install modin[ray]
from gensim.models.word2vec import FAST_VERSION
print(FAST_VERSION)
import multiprocessing
import numpy as np
# import pandas as pd
import modin
import modin.pandas as pd
from matplotlib import pyplot as plt
import os
import string

import nltk
import gensim

from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from gensim.test.utils import common_corpus, common_dictionary
from gensim.similarities import MatrixSimilarity

from gensim.test.utils import datapath, get_tmpfile
from gensim.similarities import Similarity
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
print("Cols names: {}".format(meta.columns))
meta.head(7)
plt.figure(figsize=(20,10))
meta.isna().sum().plot(kind='bar', stacked=True)
meta_dropped = meta.drop(['Microsoft Academic Paper ID', 'WHO #Covidence'], axis = 1)
plt.figure(figsize=(20,10))

meta_dropped.isna().sum().plot(kind='bar', stacked=True)
miss = meta['abstract'].isna().sum()
print("The number of papers without abstracts is {:0.0f} which represents {:.2f}% of the total number of papers".format(miss, 100* (miss/meta.shape[0])))
abstracts_papers = meta[meta['abstract'].notna()]
print("The total number of papers is {:0.0f}".format(abstracts_papers.shape[0]))
missing_doi = abstracts_papers['doi'].isna().sum()
print("The number of papers without doi is {:0.0f}".format(missing_doi))
missing_url = abstracts_papers['url'].isna().sum()
print("The number of papers without url is {:0.0f}".format(missing_url))
abstracts_papers = abstracts_papers[abstracts_papers['publish_time'].notna()]
abstracts_papers['year'] = pd.DatetimeIndex(abstracts_papers['publish_time']).year
missing_url_data = abstracts_papers[abstracts_papers["url"].notna()]
print("The total number of papers with abstracts, urls, but missing doi = {:.0f}".format( missing_url_data.doi.isna().sum()))
abstracts_papers = abstracts_papers[abstracts_papers["url"].notna()]
porter = PorterStemmer()
lancaster=LancasterStemmer()

abstracts_only = abstracts_papers['abstract']
tokenized_abs = []

for abst in abstracts_only:
    tokens_without_stop_words = remove_stopwords(abst)
    tokens_cleaned = sent_tokenize(tokens_without_stop_words)
    words = [porter.stem(w.lower()) for text in tokens_cleaned for w in word_tokenize(text) if (w.translate(str.maketrans('', '', string.punctuation))).isalnum()]
    tokenized_abs.append(words)
dictionary = []
dictionary = gensim.corpora.Dictionary(tokenized_abs)
corpus = [dictionary.doc2bow(abstract) for abstract in tokenized_abs]
tf_idf = gensim.models.TfidfModel(corpus)
def query_tfidf(query):
    
    query_without_stop_words = remove_stopwords(query)
    tokens = sent_tokenize(query_without_stop_words)

    query_doc = [porter.stem(w.lower()) for text in tokens for w in word_tokenize(text) if (w.translate(str.maketrans('', '', string.punctuation))).isalnum()]

    # mapping from words into the integer ids
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    
    return query_doc_tf_idf
def rankings(query):
    t = time.time()
    query_doc_tf_idf = query_tfidf(query)
    print(time.time()-t , 0)
    t = time.time()
    index_temp = get_tmpfile("index")
    print(time.time()-t , 1)
    t = time.time()
    len_dictionary = len(dictionary)
    print(time.time()-t , 2)
    t = time.time()
    index = Similarity(index_temp, tf_idf[corpus], num_features=len_dictionary)
    print(time.time()-t , 3)
    t = time.time()
    similarities = index[query_doc_tf_idf]
    print(time.time()-t , 4)

    # Storing similarity in the dataframe and sort from high to low simmilatiry
    t = time.time()
    abstracts_papers["similarity"] = similarities
    print(time.time()-t , 5)
    t = time.time()
#     abstracts_papers_sorted = abstracts_papers.sort_values(by ='similarity' , ascending=False)
    abstracts_papers_sorted = sorted(abstracts_papers, reverse=True)
    print(time.time()-t , 6)
    t = time.time()
    abstracts_papers_sorted.reset_index(inplace = True)
    print(time.time()-t , 7)
    t = time.time()
    top20 = abstracts_papers_sorted.head(20)
    print(time.time()-t , 8)
    t = time.time()
    norm_range = top20['year'].max() - top20['year'].min()
    print(time.time()-t , 9)
    t = time.time()
    top20["similarity"] -= (abs(top20['year'] - top20['year'].max()) / norm_range)*0.1
    print(time.time()-t , 10)
    t = time.time()
    top20 = top20.sort_values(by ='similarity' , ascending=False)
    print(time.time()-t , 11)
    t = time.time()
    top20.reset_index(inplace = True)
    print(time.time()-t , 12)
    
    return top20
import time

# query = "COVID-19 (corona) non-pharmaceutical interventions, Methods to control the spread in communities, barriers to compliance and how these vary among different populations"
t = time.time()
top = rankings(input())
print(time.time()-t)

for abstract in range(10):
    print(top.abstract[abstract])
    print('\n>>>>>>>>>>>>>>>>>>>>>>\n')
for paper in range(10):
    print(top.url[paper])