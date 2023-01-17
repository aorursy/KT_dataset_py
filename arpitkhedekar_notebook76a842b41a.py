# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv("/kaggle/input/personal-loan/Bank_Personal_Loan_Modelling-1.xlsx")
df.shape
df.head(2)
df.describe()
df.info()
df.drop(['ID'],  axis=1, inplace=True)
# X AND Y
X = df.drop(['Personal Loan'], axis=1)
Y = df['Personal Loan']
X.info()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
#MODEL TRAINING
#Gaussian Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

clf_gnb= GaussianNB()
clf_gnb.fit(X, Y)

y_pred_gnb = clf_gnb.predict(X_test)

print("Train score-" , clf_gnb.score(X_train, Y_train )*100)
print("Test scoe-", clf_gnb.score(X_test, Y_test)*100)
# K Nearest Neighbour Classifier

from sklearn.neighbors import KNeighborsClassifier
    
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, Y_train)

y_pred_knn = clf_knn.predict(X_test)

print('Train score-', clf_knn.score(X_train, Y_train)*100)
print('Test score-', clf_knn.score(X_test, Y_test)*100)


#Logistic Regression

from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()
clf_lr.fit (X_train, Y_train)

y_pred_lr = clf_lr.predict(X_test)

print('Train score-', clf_lr.score(X_train, Y_train)*100)
print('Test score-', clf_lr.score(X_test, Y_test)*100)