#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Tue Sep 25 17:07:39 2018
@author: Team_CUDS
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

test = pd.read_table('../input/test.csv',delimiter=',')
train = pd.read_table('../input/train.csv',delimiter=',')
le=LabelEncoder()
train["Cabin"] = le.fit_transform(train["Cabin"].fillna('0'))
train["Embarked"] = le.fit_transform(train["Embarked"].fillna('0'))
train["Name"] = le.fit_transform(train["Name"].fillna('0'))
train["Sex"] = le.fit_transform(train["Sex"].fillna('0'))
train["Ticket"] = le.fit_transform(train["Ticket"].fillna('0'))
train["Age"]=train["Age"].fillna('0')   
test["Cabin"] = le.fit_transform(test["Cabin"].fillna('0'))
test["Embarked"] = le.fit_transform(test["Embarked"].fillna('0'))
test["Name"] = le.fit_transform(test["Name"].fillna('0'))
test["Sex"] = le.fit_transform(test["Sex"].fillna('0'))
test["Ticket"] = le.fit_transform(test["Ticket"].fillna('0'))
test["Age"]=test["Age"].fillna('0') 
data=train.loc[:, train.columns != 'Survived']
labels = train.iloc[:,1]
# Split data into training and testing datasets
x_train, x_test1, y_train, y_test1 = train_test_split(data, labels, test_size=0.2)

# print( y_test)
# scale the data within [0-1] range
scalar = MinMaxScaler()
x_train = scalar.fit_transform(x_train)
labels = train.iloc[0:len(test),1]
x_train1, x_test, y_train1, y_test = train_test_split(test, labels, test_size=0.2)
# print(y_test)
x_test = scalar.transform(x_test)
# SVC with RBF kernel
print('Classification using SVM')
rbf_svc = svm.SVC(kernel = 'rbf', C = 1, gamma='auto')
rbf_svc.fit(x_train, y_train)
svc_pred_lab = rbf_svc.predict(x_test)

# model accuracy for testing 
accuracy = rbf_svc.score(x_test, y_test)
print('accuracy', accuracy)
# creating a confusion matrix
cm = confusion_matrix(y_test, svc_pred_lab)
print('confusion matrix')
print(cm)
#classification using KNN
print('Classification using KNN')
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, y_train)
knn_pred_lab = knn.predict(x_test)

# accuracy on X_test
accuracy = knn.score(x_test, y_test)
print('accuracy', accuracy)
# creating a confusion matrix
cm = confusion_matrix(y_test, knn_pred_lab)
print('confusion matrix')
print(cm)
#classification using Random Forest Classifier
print('Classification using Random Forest Classifier')
rfc = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
rfc.fit(x_train, y_train)
rfc_pred_lab = rfc.predict(x_test)

# accuracy on X_test
accuracy = rfc.score(x_test, y_test)
print('accuracy', accuracy)
# creating a confusion matrix
rfc_predictions = rfc.predict(x_test) 
cm = confusion_matrix(y_test, rfc_pred_lab)
print('confusion matrix')
print(cm)