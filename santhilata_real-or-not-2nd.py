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
# Read data and fill null with a value

train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df= pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



train_df = train_df.fillna('Nothing')



test_df = test_df.fillna('Nothing')
# NA data

train_df.isnull().sum()
train_df.duplicated().sum()
print(train_df.shape, test_df.shape)
print(train_df.nunique())
print(test_df.nunique())
print(train_df.keyword.value_counts())
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





print(train_df['text'].apply(lambda x: len(x.split(' '))).sum())

print(test_df['text'].apply(lambda x: len(x.split(' '))).sum())
# Clean data or Pre processing

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]') # not to remove '#' symbols 

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

    

train_df['text'] = train_df['text'].apply(clean_text)

test_df['text'] = test_df['text'].apply(clean_text)
print(train_df.loc[0]['text'], train_df.loc[0]['target'])
import seaborn as sns

sns.countplot(y=train_df.target)
def add_all_columns(x):

    return (x.keyword +' '+x['location']+' '+x['text'])

train_df['new_col'] = train_df.apply(lambda x: add_all_columns(x), axis = 1)
test_df['new_col'] = test_df.apply(lambda x: add_all_columns(x),axis=1)
train_df.head()
# Create model

X = train_df['new_col']

y = train_df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)


print(X_train.shape, y_train.shape)
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer



nb = Pipeline([('vect', CountVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', MultinomialNB()),

              ])

nb.fit(X_train.transpose(), y_train)



%time

from sklearn.metrics import classification_report

y_pred = nb.predict(X_test)



print(y_pred)

print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))
sub_pred = nb.predict(test_df.new_col) # predict test values
sub_df = pd.DataFrame({'id':test_df.id, 'target':sub_pred})

print(sub_df.target.value_counts())

sub_df.to_csv('submission1.csv', index=False)