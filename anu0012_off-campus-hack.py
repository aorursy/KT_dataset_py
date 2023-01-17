import keras

import keras.backend as K

from keras.layers import Dense, GlobalAveragePooling1D, Embedding,GlobalMaxPooling1D

from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

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

import seaborn as sns

np.random.seed(25)
train = pd.read_csv('../input/offcampus_training.csv')

test = pd.read_csv('../input/offcampus_test.csv')
train.head()
sns.countplot(train['category'])
train.isnull().sum()
train.dropna(inplace=True)
y = to_categorical(train['category'])
train_sequences = []

for txt in train.text:

    seq = []

    for i in txt.split(' '):

        seq.append(int(i))

    train_sequences.append(seq)

    

test_sequences = []

for txt in test.text:

    seq = []

    for i in txt.split(' '):

        seq.append(int(i))

    test_sequences.append(seq)
train_data = pad_sequences(train_sequences, padding="post", truncating="post", value=0,maxlen=500)

test_data = pad_sequences(test_sequences, padding="post", truncating="post", value=0, maxlen=500)



nb_words = (np.max(train_data) + 1)
from keras.layers.recurrent import LSTM, GRU

model = Sequential()

model.add(Embedding(nb_words,500,input_length=500))

model.add(GlobalMaxPooling1D())

model.add(Dense(100, activation='relu'))

model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(train_data, y, validation_split=0.2, nb_epoch=100, batch_size=32)
pred = model.predict(test_data)

pred = pred.argmax(axis=-1)

pred[:10]
result = pd.DataFrame()

result['id'] = test['id']

result['category'] = pred

result.to_csv("submission.csv", index=False)
result.head()