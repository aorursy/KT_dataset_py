# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import csv, sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

a = pd.read_csv("../input/creditcard.csv")
# break into classes

from sklearn.model_selection import train_test_split 

a = a.as_matrix()

X = a[:,:-1]

y = a[:,-1]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import SGDClassifier

from sklearn import metrics



model_hinge = SGDClassifier(loss = 'hinge', penalty = 'l2')

model_hinge = model_hinge.fit(X_train, y_train)

ypred = model_hinge.predict(X_test)

avg = 0.0

n = 200

for i in range(n):

    avg = avg + model_hinge.score(X_test,y_test)

print(avg / n)
print(metrics.classification_report(y_test, ypred))
model = SGDClassifier(loss = 'log', penalty = 'l2')

model = model.fit(X_train, y_train)

model.score(X_test,y_test)



avg = 0.0

n = 200

for i in range(n):

    avg = avg + model.score(X_test,y_test)

print(avg / n)
model = SGDClassifier(loss = 'modified_huber', penalty = 'l2')

model = model.fit(X_train, y_train)

model.score(X_test,y_test)



avg = 0.0

n = 200

for i in range(n):

    avg = avg + model.score(X_test,y_test)

print(avg / n)
from sklearn import linear_model

reg = linear_model.Ridge (alpha = .5)

reg.fit(X_train, y_train).score(X_test, y_test)