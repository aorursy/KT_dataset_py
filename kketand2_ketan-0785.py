import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.cluster import KMeans
from sklearn import preprocessing

import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;
from nltk.stem import PorterStemmer,SnowballStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("data.csv")
df = data.copy()
stemmer = SnowballStemmer('english')
df["clean_txt"] = df["text"].str.lower().apply(lambda x: re.sub(r"[-.?!\/@#_*,:;()<>|0-9\.]+",'',x))
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))            
    return result
processed_docs = df['clean_txt'].map(preprocess)
dictionary = gensim.corpora.Dictionary(processed_docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 5, 
                                   id2word = dictionary,                                    
                                   passes = 20,
                                   workers = 2)
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=4)

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
def topic_assign(unseen_document):
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    topic = []
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        #print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 10)))
        topic.append(lda_model.print_topic(index, 5))
    return topic
df["topic"] = df.clean_txt.apply(lambda x: "".join(topic_assign(x))[:65])
uniquelist = df["topic"].unique().tolist()
uniquelist
df.replace({'topic': {uniquelist[4]: "glassdoor_reviews", uniquelist[3]: "sports_news",
                      uniquelist[2]: "tech_news",uniquelist[1]: "room_rentals",
                      uniquelist[0]: "Automobiles"}}, inplace = True)
df = df[["Id","topic"]]
#df.to_csv("ketan_750_submission2.csv", index=False)
df.to_csv("ketan_750_submission3.csv", index=False)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
def topic_assign_tfidf(unseen_document):
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    corpus_tfidf = tfidf[bow_vector]
    topic = []
    for index, score in sorted(lda_model_tfidf[corpus_tfidf], key=lambda tup: -1*tup[1]):
        #print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 10)))
        topic.append(lda_model_tfidf.print_topic(index, 5))
    return topic
df["topic"] = df.clean_txt.apply(lambda x: "".join(topic_assign(x))[:65])
df
lda_train2 = gensim.models.ldamulticore.LdaMulticore(
                           corpus=bow_corpus,
                           num_topics=5,
                           id2word=dictionary,
                           #chunksize=100,
                           workers=7, # Num. Processing Cores - 1
                           passes=50,
                           eval_every = 1,
                           per_word_topics=True)
for idx, topic in lda_train2.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
def get_topic(unseen_document):
    bow_corpus = dictionary.doc2bow(unseen_document.split())
    tpk= lda_train2.get_document_topics(bow_corpus)
    tpk.sort(key=lambda x:x[1], reverse=True)
    return tpk[0][0]
get_topic(df.clean_txt[0])
df["topic"] = df.clean_txt.apply(lambda x: get_topic(x))
df["topic"] = df["topic"].astype('str')
df.replace({'topic': {"0": "Automobiles", "2": "sports_news",
                      "1": "tech_news","4": "room_rentals",
                      "3": "glassdoor_reviews"}}, inplace = True)
df[["Id","topic"]].to_csv("submission4_ketan_0785.csv", index =False)
