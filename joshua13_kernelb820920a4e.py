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
df = pd.read_csv(r'../input/heart.csv')
print(df)
dataset =  np.array(df.values)

np.random.shuffle(dataset)

print(dataset)
from sklearn import svm

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

%matplotlib inline  
X = df.iloc[:, :-1]

y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
clf = svm.LinearSVC(penalty='l1',dual=False,tol=1e-3)

clf.fit(X_train, y_train) 
prediction = clf.predict(X_test)

print(accuracy_score(y_test, prediction))
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)
prediction = clf.predict(X_test)

print(accuracy_score(y_test, prediction))
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=2)

neigh.fit(X_train,y_train) 
prediction = neigh.predict(X_test)

print(accuracy_score(y_test, prediction))
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(accuracy_score(y_test, y_pred))