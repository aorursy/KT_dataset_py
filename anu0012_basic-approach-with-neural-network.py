# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression,SGDClassifier

from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import PorterStemmer

import nltk

from nltk import word_tokenize, ngrams

from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS

import xgboost as xgb

np.random.seed(25)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
train.head()
# Target Mapping

mapping_target = {'EAP':0, 'HPL':1, 'MWS':2}

train = train.replace({'author':mapping_target})
train.head()
test_id = test['id']

target = train['author']
# function to clean data

import string

import itertools 

import re

from nltk.stem import WordNetLemmatizer

from string import punctuation



def preprocess(text):

    text = text.strip()

    text = text.replace("' ", " ' ")

    signs = set(',.:;"?!')

    prods = set(text) & signs

    if not prods:

        return text



    for sign in prods:

        text = text.replace(sign, ' {} '.format(sign) )

    return text



def cleanData(text, lowercase = False, remove_stops = False, stemming = False, lemmatization = False):

    

    txt = str(text)

    

    txt = re.sub(r'[^A-Za-z\s]',r' ',txt)

    

     

    if lowercase:

        txt = " ".join([w.lower() for w in txt.split()])

        

    if remove_stops:

        txt = " ".join([w for w in txt.split() if w not in stops])

    if stemming:

        st = PorterStemmer()

        txt = " ".join([st.stem(w) for w in txt.split()])

    

    if lemmatization:

        wordnet_lemmatizer = WordNetLemmatizer()

        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])



    return txt
train['text'] = train['text'].map(lambda x: preprocess(x))

test['text'] = test['text'].map(lambda x: preprocess(x))
# clean text

train['text'] = train['text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = False))

test['text'] = test['text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = False))
np.random.seed(25)

from keras.models import Sequential

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten, merge, LSTM, Lambda, Dropout

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model

from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras.layers.normalization import BatchNormalization

from keras import backend as K

from keras.layers import Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.layers.merge import concatenate

from keras.layers.core import Dense, Activation, Dropout

import codecs
MAX_SEQUENCE_LENGTH = 256

MAX_NB_WORDS = 200000
def add_ngram(q, n_gram_max):

            ngrams = []

            for n in range(2, n_gram_max+1):

                for w_index in range(len(q)-n+1):

                    ngrams.append('--'.join(q[w_index:w_index+n]))

            return q + ngrams
n_gram_max = 2

print('Processing text dataset')

texts_1 = []

for text in train['text']:

    text = text.split()

    texts_1.append(' '.join(add_ngram(text, n_gram_max)))

    



labels = train['author']  # list of label ids



print('Found %s texts.' % len(texts_1))

test_texts_1 = []

for text in test['text']:

    text = text.split()

    test_texts_1.append(' '.join(add_ngram(text, n_gram_max)))

print('Found %s texts.' % len(test_texts_1))
min_count = 2

tokenizer = Tokenizer(lower=False, filters='')

tokenizer.fit_on_texts(texts_1)

num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])



tokenizer = Tokenizer(num_words=num_words, lower=True, filters='')

tokenizer.fit_on_texts(texts_1)





sequences_1 = tokenizer.texts_to_sequences(texts_1)

# word_index = tokenizer.word_index

# print('Found %s unique tokens.' % len(word_index))



test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)



data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.array(labels)

print('Shape of data tensor:', data_1.shape)

print('Shape of label tensor:', labels.shape)



test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)

#test_labels = np.array(test_labels)

del test_sequences_1

del sequences_1

import gc

gc.collect()
nb_words = np.max(data_1) + 1 #min(MAX_NB_WORDS, len(word_index)) + 1
nb_words
from keras.layers.recurrent import LSTM, GRU

model = Sequential()

model.add(Embedding(nb_words,20,input_length=MAX_SEQUENCE_LENGTH))

# model.add(Flatten())

# model.add(Dense(100, activation='relu'))

# model.add(Dropout(0.3))

# model.add(Conv1D(64,

#                  5,

#                  padding='valid',

#                  activation='relu'))

# model.add(Dropout(0.3))

model.add(GlobalAveragePooling1D())

# model.add(Flatten())

# model.add(Dense(100, activation='relu'))

# model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(data_1, to_categorical(labels), validation_split=0.2, nb_epoch=15, batch_size=16)
preds = model.predict(test_data_1)
result = pd.DataFrame()

result['id'] = test_id

result['EAP'] = [x[0] for x in preds]

result['HPL'] = [x[1] for x in preds]

result['MWS'] = [x[2] for x in preds]



result.to_csv("result.csv", index=False)
result.head()