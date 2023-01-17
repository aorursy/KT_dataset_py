# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.DataFrame()
dftr = pd.read_csv('../input/train.csv')
dfte = pd.read_csv('../input/test.csv')
print(dftr.columns)
#print(dfte.columns)
#print(dfte.isnull().sum())
le = LabelEncoder()
dftr['Sex'] = le.fit_transform(dftr['Sex'])
median1 = dftr['Age'].median()
dftr['Age'] = dftr['Age'].fillna(median1)
dftr = dftr.set_index(dftr['PassengerId'])
del dftr['PassengerId']
del dftr['Cabin']
del dftr['Name']
del dftr['Ticket']
dftr = dftr.dropna()
dftr['Embarked'] = le.fit_transform(dftr['Embarked'])
y = dftr['Survived']
del dftr['Survived']
print(dftr.isnull().sum())
#print(len(dftr),len(y))
print(dfte.columns)
#print(dfte.columns)
print(len(dfte))
le = LabelEncoder()
dfte['Embarked'] = le.fit_transform(dfte['Embarked'])
dfte['Sex'] = le.fit_transform(dfte['Sex'])
mean = dfte['Fare'].mean()
dfte['Fare'] = dfte['Fare'].fillna(mean)
median2 = dfte['Age'].median()
dfte['Age'] = dfte['Age'].fillna(median2)
dfte = dfte.set_index(dfte['PassengerId'])
del dfte['PassengerId']
del dfte['Cabin']
del dfte['Name']
del dfte['Ticket']
dfte = dfte.dropna()
print(dfte.isnull().sum())
#print(len(dfte))

X1_train = dftr.copy()
X1_test = dfte.copy()
y1_train = y.copy()
#print(len(X1_train),len(y1_train))
yt = pd.read_csv('../input/gender_submission.csv')
del yt['PassengerId']
#print(len(X1_test),len(yt))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 1, penalty = 'l1')
lr.fit(X1_train,y1_train)
prd1 = lr.predict(X1_test)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X1_train,y1_train)
prd2 = clf.predict(X1_test)
from sklearn.svm import SVC
sv = SVC(C= 100, gamma = 'auto')
sv.fit(X1_train,y1_train)
prd3 = sv.predict(X1_test)
from sklearn import metrics
print('------------------------')
#print('Logistic Regression Score Train data:',metrics.r2_score(y1_train,X1_train))
print('Logistic Regression Score Test data:',metrics.r2_score(yt,prd1))
print('------------------------')
#print('Random Forest Classifier Score Train data:',metrics.r2_score(y1_train,X1_train))
print('Random Forest Classifier Score Test data:',metrics.r2_score(yt,prd2))
print('------------------------')
#print('Random Forest Classifier Score Train data:',metrics.r2_score(y1_train,X1_train))
print('Support vector Classifier Score Test data:',metrics.r2_score(yt,prd3))