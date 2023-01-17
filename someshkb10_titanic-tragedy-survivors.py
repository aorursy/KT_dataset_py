# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#import pandas as pd
#import numpy as np
#
import matplotlib.pyplot as plt
#
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

file = pd.read_csv('../input/train.csv')

file2 = pd.read_csv('../input/test.csv')

df_train = pd.DataFrame(file)

df_test = pd.DataFrame(file2)

#print(df_train.describe())
#print(df_train.info()) 

#print(df_test.describe())
#print(df_test.info())

le = LabelEncoder()

df_train['Sex']  = le.fit_transform(df_train['Sex'])
df_test['Sex'] = le.fit_transform(df_test['Sex'])


df_train['Embarked'] = df_train['Embarked'].fillna('')#(method='ffill',inplace=True)
df_train['Embarked'] = le.fit_transform(df_train['Embarked'])
df_test['Embarked'] = le.fit_transform(df_test['Embarked'])

df_train['Age'] = df_train['Age'].fillna(method='ffill') 
df_test['Age'] = df_test['Age'].fillna(method='ffill')
df_test['Fare'] = df_test['Fare'].fillna(method='ffill')
#print(df_train['Age'])

#print(df_test.info())

X_train = df_train.drop(['PassengerId','Survived','Name','Ticket','Cabin'],axis=1)

X_test = df_test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
#print(X_train.info())
#print(X_test.info())
Y_train = df_train['Survived']
#print(Y_train)


print('-----------------------------------')
print('----Use KNeighbor Classification---')
print('-----------------------------------')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
print(('score =') + str(round(knn.score(X_train,Y_train)*100,2)))

print('-----------------------------------')
print('----Use Logistic Regression--------')
print('-----------------------------------')
            
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred = logreg.predict(X_test)
print(('score =') + str(round(logreg.score(X_train,Y_train)*100,2)))

print('-----------------------------------')
print('----Use DecisionTree --------------')
print('-----------------------------------')

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print(('score =') + str(round(clf.score(X_train,Y_train)*100,2)))

print('-----------------------------------')
print('----Use Support Vector Machine-----')
print('-----------------------------------')

clf = svm.LinearSVC()
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print(('score =') + str(round(clf.score(X_train,Y_train)*100,2)))
# Any results you write to the current directory are saved as output.
