import os

import re

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

import json

import time

import warnings 

warnings.filterwarnings('ignore')

import datetime



df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

print(df.shape)
df.info()

col='abstract'

keep = df.dropna(subset=[col])

print(keep.shape)

docs = keep[col].tolist()
# Code adaptead from https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html



from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

from gensim.corpora import Dictionary



tokenizer = RegexpTokenizer(r'\w+')

for idx in range(len(docs)):

    # Convert to lowercase.

    docs[idx] = docs[idx].lower()  

    # Split into words.

    docs[idx] = tokenizer.tokenize(docs[idx])  



# Remove numbers

docs = [[token for token in doc if not token.isnumeric()] for doc in docs]



# Remove one-character words

docs = [[token for token in doc if len(token) > 1] for doc in docs]



# Remove stopwords 

stop_words = stopwords.words("english")

docs = [[token for token in doc if token not in stop_words] for doc in docs]



# Lemmatize

lemmatizer = WordNetLemmatizer()

docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]



# Create a dictionary representation of the documents

dictionary = Dictionary(docs)



# Filter out words that occur less than 20 documents, or more than 50% of the documents

dictionary.filter_extremes(no_below=20, no_above=0.5)



# Create Bag-of-words representation of the documents

corpus = [dictionary.doc2bow(doc) for doc in docs]



print('Number of unique tokens: %d' % len(dictionary))

print('Number of documents: %d' % len(corpus))
from gensim.models import LdaModel, LdaMulticore



# Set training parameters.

num_topics = 10



# Make a index to word dictionary.

temp = dictionary[0]  # This is only to "load" the dictionary.

id2word = dictionary.id2token



model = LdaMulticore(

    corpus=corpus,

    id2word=id2word,

    chunksize=2000,

    eta='auto',

    iterations=10,

    num_topics=num_topics,

    passes=10,

    eval_every=None,

    workers=4

)
top_topics = model.top_topics(corpus) 

for i, (topic, sc) in enumerate(top_topics): 

    print("\nTopic {}: ".format(i) + ", ".join([w for score,w in topic]))
def split_to_sentences(text):

    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
has_question = keep[keep.abstract.dropna().map(lambda x: '?' in x)]

has_question.shape
df.publish_time.dropna().map(lambda x: x.split()[0].split('-')[0]).value_counts()
for i, row in has_question.dropna(subset=['publish_time']).iterrows():

    if not '2020' in row.publish_time:

        continue

    print("\nTITLE: {}".format(row.title))

    print("\tDATE:  {}".format(row.publish_time))

    for sent in split_to_sentences(row.abstract):

        if sent[-1] == '?':

            print("Q:\t{}".format(sent))