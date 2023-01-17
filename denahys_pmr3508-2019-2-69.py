import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
df.describe()
df.columns
df.dropna(inplace=True)

df.drop(axis=1, columns="Id", inplace=True)
df.head()
from sklearn import preprocessing

qualiVars = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

df[qualiVars] = df[qualiVars].apply(preprocessing.LabelEncoder().fit_transform)
Xtrain = df.iloc[:,0:-1]

Ytrain = df.income
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtrain,Ytrain,test_size=0.20)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print("Random forest metrics")

print(confusion_matrix(y_test, rfc_pred))

print(classification_report(y_test, rfc_pred))

print('Accuracy: ',accuracy_score(y_test, rfc_pred))
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
log_pred = logmodel.predict(X_test)
print("Logistic regression metrics")

print(confusion_matrix(y_test, log_pred))

print(classification_report(y_test, log_pred))

print('Accuracy: ',accuracy_score(y_test, log_pred))
from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train,y_train)
svm_pred = svm.predict(X_test)
print("SVM metrics")

print(confusion_matrix(y_test, svm_pred))

print(classification_report(y_test, svm_pred))

print('Accuracy: ',accuracy_score(y_test, svm_pred))