# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



# Any results you write to the current directory are saved as output.



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
data_text.iloc[1]['Text']
from nltk import word_tokenize

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

stop.update(['href','br'])

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')



for idx in range(len(data_text)):

    data_text.iloc[idx]['Text'] = [word for word in tokenizer.tokenize(data_text.iloc[idx]['Text'].lower()) if word not in stop]

    
train_ = [value[0] for value in data_text.iloc[0:].values]

num_topics = 8
id2word = gensim.corpora.Dictionary(train_)

corpus = [id2word.doc2bow(text) for text in train_]

lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
def get_lda_topics(model, num_topics):

    word_dict = {};

    for i in range(num_topics):

        words = model.show_topic(i, topn = 200);

        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];

    return pd.DataFrame(word_dict)
get_lda_topics(lda, num_topics)