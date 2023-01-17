

import pandas as pd

import os

import re

import spacy

from sklearn import preprocessing

from gensim.models.phrases import Phrases, Phraser

from time import time 

import multiprocessing

from gensim.models import Word2Vec

import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool

from bokeh.plotting import figure, show, output_notebook

from sklearn.manifold import TSNE

import numpy as np

from sklearn.preprocessing import scale

import keras 

from keras.models import Sequential, Model 

from keras import layers

from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Embedding

from keras.layers.merge import Concatenate

from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud

from nltk.tokenize import RegexpTokenizer

from sklearn.metrics import confusion_matrix

from nltk.tokenize import RegexpTokenizer

import matplotlib.pyplot as plt

import gensim

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
for name in train.columns:

    x = train[name].isna().sum()

    if x > 0:

        val_list = np.random.choice(train.groupby(name).count().index, x, p=train.groupby(name).count()['id'].values /sum(train.groupby(name).count()['id'].values))

        train.loc[train[name].isna(), name] = val_list
for name in test.columns:

    x = test[name].isna().sum()

    if x > 0:

        val_list = np.random.choice(test.groupby(name).count().index, x, p=test.groupby(name).count()['id'].values /sum(test.groupby(name).count()['id'].values))

        test.loc[test[name].isna(), name] = val_list
train_df = train.drop('target',axis = 1)
data = pd.concat([train_df,test])
le = preprocessing.LabelEncoder()

for name in data.columns:

    if name == 'keyword' or name == 'location':

        print(name)

        data[name] = data[name].astype(str)

        train[name] = train[name].astype(str)

        test[name] = test[name].astype(str)

        le.fit(data[name])

        train[name] = le.transform(train[name])

        test[name] = le.transform(test[name])
tweets_train = train['text'].values

y = train['target'].values

tweets_test = test['text'].values
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(binary=True, use_idf=True)

tfidf_train_data = vec.fit_transform(tweets_train) 

tfidf_test_data = vec.transform(tweets_test)
from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression()

classifier.fit(tfidf_train_data, y)

score = classifier.score(tfidf_train_data, y)



print("Accuracy:", score)
predictions = classifier.predict(tfidf_test_data)
len(predictions)
sample_submission['target'] = predictions
sample_submission.head(60)
sample_submission.to_csv('submission.csv',index = False)