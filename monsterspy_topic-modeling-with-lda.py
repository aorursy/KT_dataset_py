import pandas as pd

import numpy as np

import scipy as sp

import sklearn

import sys

from nltk.corpus import stopwords

import nltk

from gensim.models import ldamodel

import gensim.corpora;

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.preprocessing import normalize

import pickle

import re
data = pd.read_csv('../input/Reviews.csv')
data = data.sample(n = 10000)
data_text = data[['Text']]
data_text.iloc[0]['Text']
from nltk import word_tokenize

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

stop.update(['href','br'])

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')



for idx in range(len(data_text)):

    data_text.iloc[idx]['Text'] = [word for word in tokenizer.tokenize(data_text.iloc[idx]['Text'].lower()) if word not in stop]
train_ = [value[0] for value in data_text.iloc[0:].values]

num_topics = 5
id2word = gensim.corpora.Dictionary(train_)

corpus = [id2word.doc2bow(text) for text in train_]

lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
def get_lda_topics(model, num_topics):

    word_dict = {};

    for i in range(num_topics):

        words = model.show_topic(i, topn = 40);

        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];

    return pd.DataFrame(word_dict)
get_lda_topics(lda, num_topics)