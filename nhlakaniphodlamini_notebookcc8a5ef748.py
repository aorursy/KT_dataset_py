# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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

from sklearn import datasets, linear_model, metrics



from sklearn.metrics import f1_score



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/nhlakanipho-dlamini-climate-change/train.csv')

test = pd.read_csv('../input/nhlakanipho-dlamini-climate-change/test.csv')
train.sentiment.value_counts()
y = train['sentiment']

X = train['message']
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words="english")

X_vectorized = vectorizer.fit_transform(X)
X_train,X_val,y_train,y_val = train_test_split(X_vectorized,y,test_size=26,random_state=66)
Lrsvc = LinearSVC()

Lrsvc.fit(X_train, y_train)

Lrsvc_pred = Lrsvc.predict(X_val)
f1_score(y_val, Lrsvc_pred, average="macro")
testx = test['message']

test_vect = vectorizer.transform(testx)
y_pred = Lrsvc.predict(test_vect)
test['sentiment'] = y_pred
test.head()
test[['tweetid','sentiment']].to_csv('testsubmission_Nhlakanipho_Dlamini_final_5.csv', index=False)