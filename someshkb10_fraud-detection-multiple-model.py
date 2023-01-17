# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print ('Credit Card Fraud Detection')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm 

from sklearn.metrics import classification_report 

file=pd.read_csv('../input/creditcard.csv') 

df = pd.DataFrame(file)

#print(df.info())
#print(df.describe()) 

X = df.iloc[:,0:29]

Y = df.iloc[:,30]



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

print(Y_test.value_counts())

print('---------------------------------------------')
print('-------Logistic Regression-------------------')
print('---------------------------------------------')
lr = LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred_lr = lr.predict(X_test)
print('score = ' + str(lr.score(X_train,Y_train))) 
print(str(classification_report(Y_test, Y_pred_lr)))

#print(Y_test)
#print(Y_pred_lr)

#sns.swarmplot(Y_test,Y_pred)
#sns.swarmplot(X_test,Y_test)
#plt.show()
print('---------------------------------------------')
print('-------------Decision Tree-------------------')
print('---------------------------------------------')

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,Y_train)
Y_pred_dt = clf.predict(X_test)
print('score = ' + str(clf.score(X_train,Y_train)))
print(str(classification_report(Y_test, Y_pred_dt)))

print('---------------------------------------------')
print('-------------Random Forest-------------------')
print('---------------------------------------------')
model = RandomForestClassifier()
model.fit(X_train,Y_train)
Y_pred_rf = model.predict(X_test)
print('score = ' + str(model.score(X_train,Y_train)))
print(str(classification_report(Y_test, Y_pred_rf)))

print('-----------------------------------')
print('----Use Support Vector Machine-----')
print('-----------------------------------')

clf = svm.LinearSVC()
clf.fit(X_train,Y_train)
Y_pred_svm = clf.predict(X_test)
print(('score =') + str(round(clf.score(X_train,Y_train))))
print(str(classification_report(Y_test, Y_pred_svm)))

print('===== Fitting Completed ============')




# Any results you write to the current directory are saved as output.
