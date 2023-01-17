# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Logistic Regression gives 96.1 accuracy on human activity dataset



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/train.csv')

X_train= dataset.iloc[:,:-1].values

y_train= dataset.iloc[:, -1].values



dataset = pd.read_csv('../input/test.csv')

X_test= dataset.iloc[:,:-1].values

y_test= dataset.iloc[:, -1].values



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score

cm = confusion_matrix(y_test, y_pred)

accuracy_score=accuracy_score(y_test,y_pred)

recall_score=recall_score(y_test,y_pred,average='weighted')

f1_score=f1_score(y_test,y_pred,average='weighted')

print(y_pred)

print(cm)

print(accuracy_score)

print(recall_score)

print(f1_score)
