%matplotlib inline

from matplotlib import pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.tail()
test.tail()
train = train.fillna(' ')
test = test.fillna(' ')
train['article'] = train['title'] + ' ' + train['author'] + ' ' + train['text']
test['article'] = test['title'] + ' ' + test['author'] + ' ' + test['text']
train.tail()
test.tail()
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
transformer = TfidfTransformer(smooth_idf=False)
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = ngram_vectorizer.fit_transform(train['article'].values)
tfidf = transformer.fit_transform(counts)
tfidf.data
targets = train['label'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=1)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)