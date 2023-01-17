# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df.head()
test_df.head()
train_df.fillna('Nothing')

test_df.fillna('Nothing')
train = train_df[['text','target']].copy()

print(train.head())



test = test_df[['id','text']].copy()

print(test.head())
print('train.shape ',train.shape)

print('test.shape ', test.shape)
import logging

from numpy import random

import gensim

import nltk

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

import re

from bs4 import BeautifulSoup

%matplotlib inline





print(train['text'].apply(lambda x: len(x.split(' '))).sum())

print(test['text'].apply(lambda x: len(x.split(' '))).sum())
#plt.figure(figsize=(10,4))

train.target.value_counts()
def print_plot(index):

    example = train[train.index == index][['text', 'target']].values[0]

    if len(example) > 0:

        print(example[0])

        print('target:', example[1])
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]') # remove '#' symbols partcularly

STOPWORDS = set(stopwords.words('english'))



def clean_text(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = BeautifulSoup(text, "lxml").text # HTML decoding

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text

    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text

    return text

    

train['text'] = train['text'].apply(clean_text)

print_plot(10)

test['text'] = test['text'].apply(clean_text)

X = train.text

y = train.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer



nb = Pipeline([('vect', CountVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', MultinomialNB()),

              ])

nb.fit(X_train, y_train)



%time

from sklearn.metrics import classification_report

y_pred = nb.predict(X_test)



print(y_pred)

print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))
from sklearn.linear_model import SGDClassifier



sgd = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),

               ])

sgd.fit(X_train, y_train)



%time



y_pred = sgd.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))
from sklearn.linear_model import LogisticRegression



logreg = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', LogisticRegression(n_jobs=1, C=1e5)),

               ])

logreg.fit(X_train, y_train)



%time



y_pred = logreg.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))
print(test)
sub_pred = nb.predict(test.text) # predict test values
sub_df = pd.DataFrame({'id':test.id, 'target':sub_pred})
print(sub_df.target.value_counts())
#sub_df = sub_df.sort_values(by='target')
sub_df.to_csv('submission.csv', index=False)
a = pd.read_csv('submission.csv')
a
a[a.id==3267]