# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
dataset = pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
data= dataset['Review']

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(data).toarray()
X = TfidfTransformer(X)
y = dataset.iloc[:, 1].values
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
accuracy = []
mcc_value =[]
test_Y = []
predicted = []
cm =[]
skf = StratifiedKFold(n_splits = 6, shuffle = True)
for train_ix, test_ix in skf.split(data,y):
           X_train, X_test = data[train_ix], data[test_ix]
           y_train, y_test = y[train_ix], y[test_ix]
           cv = CountVectorizer(max_features = 1500)
           tfidf_vect = TfidfTransformer()
           X_train = cv.fit_transform(X_train)
           X_train = tfidf_vect.fit_transform(X_train)
           # y_train = dataset.iloc[:, 1].values
           X_test = cv.transform(X_test)
           X_test = tfidf_vect.transform(X_test)
           classifier = MultinomialNB(alpha=0.3)
           classifier.fit(X_train, y_train)
           y_pred = classifier.predict(X_test)
           ac = np.mean(y_pred == y_test)
           test_Y.append(y_test)
           predicted.append(y_pred)
           accuracy.append(ac)
           mcc_value.append(matthews_corrcoef(y_test, y_pred))
           print(ac)
           cm.append(confusion_matrix(y_test, y_pred))

avg = sum(accuracy) / len(accuracy)
print("Avg accuracy",avg)
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

accuracy = []
mcc_value =[]
test_Y = []
predicted = []
cm =[]
skf = StratifiedKFold(n_splits = 6, shuffle = True)
for train_ix, test_ix in skf.split(data,y):
           X_train, X_test = data[train_ix], data[test_ix]
           y_train, y_test = y[train_ix], y[test_ix]
           cv = CountVectorizer(max_features = 1500)
           tfidf_vect = TfidfTransformer()
           X_train = cv.fit_transform(X_train)
           X_train = tfidf_vect.fit_transform(X_train)
           # y_train = dataset.iloc[:, 1].values
           X_test = cv.transform(X_test)
           X_test = tfidf_vect.transform(X_test)
           classifier = BernoulliNB(alpha=0.8)
           classifier.fit(X_train, y_train)
           y_pred = classifier.predict(X_test)
           ac = np.mean(y_pred == y_test)
           test_Y.append(y_test)
           predicted.append(y_pred)
           accuracy.append(ac)
           mcc_value.append(matthews_corrcoef(y_test, y_pred))
           print(ac)
           cm.append(confusion_matrix(y_test, y_pred))

avg = sum(accuracy) / len(accuracy)
print("Avg accuracy",avg)
from sklearn import linear_model
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

accuracy = []
mcc_value =[]
test_Y = []
predicted = []
cm =[]
skf = StratifiedKFold(n_splits = 6, shuffle = True)
for train_ix, test_ix in skf.split(data,y):
           X_train, X_test = data[train_ix], data[test_ix]
           y_train, y_test = y[train_ix], y[test_ix]
           cv = CountVectorizer(max_features = 1500)
           tfidf_vect = TfidfTransformer()
           X_train = cv.fit_transform(X_train)
           X_train = tfidf_vect.fit_transform(X_train)
           # y_train = dataset.iloc[:, 1].values
           X_test = cv.transform(X_test)
           X_test = tfidf_vect.transform(X_test)
           classifier = linear_model.LogisticRegression(C=1.5)
           classifier.fit(X_train, y_train)
           y_pred = classifier.predict(X_test)
           ac = np.mean(y_pred == y_test)
           test_Y.append(y_test)
           predicted.append(y_pred)
           accuracy.append(ac)
           mcc_value.append(matthews_corrcoef(y_test, y_pred))
           print(ac)
           cm.append(confusion_matrix(y_test, y_pred))


avg = sum(accuracy) / len(accuracy)
print("Avg accuracy",avg)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
accuracy = []
mcc_value =[]
test_Y = []
predicted = []
cm =[]
skf = StratifiedKFold(n_splits = 6, shuffle = True)
for train_ix, test_ix in skf.split(data,y):
           X_train, X_test = data[train_ix], data[test_ix]
           y_train, y_test = y[train_ix], y[test_ix]
           cv = CountVectorizer(max_features = 1500)
           tfidf_vect = TfidfTransformer()
           X_train = cv.fit_transform(X_train)
           X_train = tfidf_vect.fit_transform(X_train)
           # y_train = dataset.iloc[:, 1].values
           X_test = cv.transform(X_test)
           X_test = tfidf_vect.transform(X_test)
           classifier = DecisionTreeClassifier()
           classifier.fit(X_train, y_train)
           y_pred = classifier.predict(X_test)
           ac = np.mean(y_pred == y_test)
           test_Y.append(y_test)
           predicted.append(y_pred)
           accuracy.append(ac)
           mcc_value.append(matthews_corrcoef(y_test, y_pred))
           print(ac)
           cm.append(confusion_matrix(y_test, y_pred))

avg = sum(accuracy) / len(accuracy)
print("Avg accuracy",avg)
from sklearn import svm
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
accuracy = []
mcc_value =[]
test_Y = []
predicted = []
cm =[]
skf = StratifiedKFold(n_splits = 6, shuffle = True)
for train_ix, test_ix in skf.split(data,y):
           X_train, X_test = data[train_ix], data[test_ix]
           y_train, y_test = y[train_ix], y[test_ix]
           cv = CountVectorizer(max_features = 1500)
           tfidf_vect = TfidfTransformer()
           X_train = cv.fit_transform(X_train)
           X_train = tfidf_vect.fit_transform(X_train)
           # y_train = dataset.iloc[:, 1].values
           X_test = cv.transform(X_test)
           X_test = tfidf_vect.transform(X_test)
           classifier = svm.SVC()
           classifier.fit(X_train, y_train)
           y_pred = classifier.predict(X_test)
           ac = np.mean(y_pred == y_test)
           test_Y.append(y_test)
           predicted.append(y_pred)
           accuracy.append(ac)
           mcc_value.append(matthews_corrcoef(y_test, y_pred))
           print(ac)
           cm.append(confusion_matrix(y_test, y_pred))

avg = sum(accuracy) / len(accuracy)
print("Avg accuracy",avg)
'''MultinomialNB - Avg accu 0.8009179575444635
BernoulliNB - Avg accuracy 0.7940571811053738
LogisticRegression - Avg accuracy 0.8180340409256073
DecisionTreeClassifier - Avg accuracy 0.7301587301587301
svm - Avg accuracy 0.7419559189137502'''