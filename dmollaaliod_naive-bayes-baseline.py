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
print(os.listdir("../input/alta-2018-challenge"))
train = pd.read_csv('../input/alta2018traindata/train_data.csv')
train.head()
train['first_ipc_mark_section'].value_counts().plot.bar()
train['first_ipc_mark_section'].value_counts()
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
first_baseline = DummyClassifier(strategy="most_frequent")
first_baseline_scores = cross_val_score(first_baseline, train[['id']], train['first_ipc_mark_section'], cv=5, scoring='f1_micro')
print(first_baseline_scores.mean())
print(first_baseline_scores.std())
train['filename'] = ['../input/alta2018patents/patents/patents/'] + train['id'].astype(str) + ['.txt']
train.head()
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
tfidf = TfidfVectorizer(input='filename', encoding='iso8859-1')
nb = naive_bayes.MultinomialNB()
second_baseline = Pipeline(steps=[('tfidf', tfidf), ('nb', nb)])

second_baseline_scores = cross_val_score(second_baseline, train['filename'], train['first_ipc_mark_section'], cv=5, scoring='f1_micro')
second_baseline_scores.mean()
second_baseline_scores.std()
second_baseline.fit(train['filename'], train['first_ipc_mark_section'])
test = pd.read_csv('../input/alta2018testdata/test_data.csv')
test.head()
test['filename'] = ['../input/alta2018patents/patents/patents/'] + test['id'].astype(str) + ['.txt']
test['first_ipc_mark_section'] = second_baseline.predict(test['filename'])
test.head()
test['first_ipc_mark_section'].value_counts().plot.bar()
test[['id', 'first_ipc_mark_section']].to_csv('test_results.csv', index=False)
