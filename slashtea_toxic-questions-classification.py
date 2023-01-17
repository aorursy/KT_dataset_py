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
train_data = pd.read_csv('../input/train.csv')
train_data.head()
train_data.info()
train_data.isnull().any()
train_data['target'].value_counts()
from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer instance.
tfidf = TfidfVectorizer()

# Feature vector and target variable.
X = train_data['question_text']
y = train_data['target'].values

X_vec = tfidf.fit_transform(X)
X_vec.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, stratify=y)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred) *100, 2)
submission_sample = pd.read_csv('../input/sample_submission.csv')
submission_sample.head()
test = pd.read_csv('../input/test.csv')
test.head()
test_vec = tfidf.transform(test['question_text'])
test_vec.shape
test_pred = clf.predict(test_vec)
test_pred
y_pred_result = pd.Series(test_pred)
y_pred_result.name = 'prediction'
submit = pd.concat([test['qid'], y_pred_result], axis=1, names=['qid', 'prediction'])
submit.to_csv('submission.csv', index=False)
