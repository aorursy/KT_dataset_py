# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import re

import nltk

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings("ignore")
from nltk.corpus import stopwords

nltk_stopwords = stopwords.words('english')
print(nltk_stopwords)
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
print(len(train))

print(len(test))
def clean_text(text):

    text = text.lower()

    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)   

    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)   

    text = re.sub(r'www.[^ ]+', '', text)  

    text = re.sub(r'[a-zA-Z0-9]*www[a-zA-Z0-9]*com[a-zA-Z0-9]*', '', text)  

    text = re.sub(r'[^a-zA-Z]', ' ', text)   

    text = [token for token in text.split() if len(token) > 2]

    text = ' '.join(text)

    return text



train['text'] = train['text'].apply(clean_text)

test['text'] = test['text'].apply(clean_text)
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(train['text'], train['sentiment'], test_size=0.25, stratify=train['sentiment'], random_state=1)
print(type(X_train))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, 

                             min_df=3, max_features=None, binary=False, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
X_train_tfidf = tfidf_vect.fit_transform(X_train)

X_cv_tfidf = tfidf_vect.transform(X_cv)
from sklearn.metrics import f1_score, accuracy_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='log', max_iter=200, random_state=0, class_weight='balanced')

ovr = OneVsRestClassifier(sgd)

ovr.fit(X_train_tfidf, y_train)

y_pred_class = ovr.predict(X_cv_tfidf)

print('f1_score       :', f1_score(y_cv, y_pred_class, average='macro'))

print('accuracy score :', accuracy_score(y_cv, y_pred_class))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, 

                             min_df=3, max_features=None, binary=False, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
full_text = list(train['text'].values) + list(test['text'].values)

tfidf_vect.fit(full_text)



X_train_tfidf = tfidf_vect.transform(train['text'])

X_test_tfidf = tfidf_vect.transform(test['text'])



y_train = train['sentiment']
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='log', max_iter=200, random_state=0, class_weight='balanced')

ovr = OneVsRestClassifier(sgd)

ovr.fit(X_train_tfidf, y_train)

y_pred_class = ovr.predict(X_test_tfidf)

y_pred_class
test['sentiment'] = y_pred_class

test.drop(['text','drug'], axis=1,inplace=True)

test.head()
test['sentiment'].value_counts()
test.to_csv('submission.csv', index=False)