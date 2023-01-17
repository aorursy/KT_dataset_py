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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
df = pd.read_csv('/kaggle/input/cancer-data-text-classification/cancer_data.tsv', sep = '\t')
df.head()
df.shape
df.columns = ['reviews', 'cat']
df.head()
df['cat'].unique()
mind = {'no':'no', 'no ': 'no', ' no':'no', 'yes':'yes', 'yes ':'yes'}
df['cat'] = [mind[x] for x in df['cat']]
df['cat'].unique()
pd.set_option('display.max_colwidth', 200)

df.head()
df['cat'].value_counts()
import nltk

import re

import string
def remove_links(tweet):

    '''Takes a string and removes web links from it'''

    tweet = re.sub(r'http\S+', '', tweet) # remove http links

    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links

    tweet = tweet.strip('[link]') # remove [links]

    return tweet
df['reviews']=df['reviews'].apply(lambda x: remove_links(x))
df.head()
def remove_users(tweet):

    '''Takes a string and removes retweet and @user information'''

    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet

    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at

    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at

    return tweet
df['reviews']=df['reviews'].apply(lambda x: remove_users(x))
df.head()
def remove_punc(text):

    no_punc = ''.join([c for c in text if c not in string.punctuation])

    return no_punc
df['reviews']=df['reviews'].apply(lambda x: remove_punc(x))
df.head()
from sklearn.model_selection import train_test_split
X = df['reviews']

y = df['cat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer
text_cal = Pipeline([('tfid', TfidfVectorizer()), ('cal', LinearSVC())])
text_cal.fit(X_train, y_train)
prediction = text_cal.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))
print(metrics.accuracy_score(y_test, prediction))