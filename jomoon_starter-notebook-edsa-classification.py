import pandas as pd

import numpy as np

import nltk

import string

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



from sklearn.metrics import f1_score



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/climate-change-edsa2020-21/train.csv')

test = pd.read_csv('../input/climate-change-edsa2020-21/test.csv')
train.sentiment.value_counts()
y = train['sentiment']

X = train['message']
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words="english")

X_vectorized = vectorizer.fit_transform(X)
X_train,X_val,y_train,y_val = train_test_split(X_vectorized,y,test_size=.3,shuffle=True, stratify=y, random_state=11)
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_val)
f1_score(y_val, rfc_pred, average="macro")
testx = test['message']

test_vect = vectorizer.transform(testx)
y_pred = rfc.predict(test_vect)
test['sentiment'] = y_pred
test.head()
test[['tweetid','sentiment']].to_csv('testsubmission.csv', index=False)