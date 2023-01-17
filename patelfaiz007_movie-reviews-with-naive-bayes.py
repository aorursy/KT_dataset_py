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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import nltk
data = pd.read_csv('../input/imdb_labelled.txt', delimiter = '\t', header = None)
data.columns = ['Review', 'Sentiment']
data.head()
data.shape
X = data.values[:,0]
y = data.values[:,1]
y = y.astype(int)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv.fit(X)
X = cv.transform(X)
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
model = BernoulliNB(alpha = 1.0)
#model = MultinomialNB(alpha = 1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
test = ['That was a horrible movie, A complete waste of Time']
test = cv.transform(test)
test_pred = model.predict(test)
print(test_pred)
# Trying MultinomialNB
#model = BernoulliNB(alpha = 1.0)
model = MultinomialNB(alpha = 1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
