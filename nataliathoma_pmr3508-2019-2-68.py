import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
df = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

df.describe()
df.columns
df.dropna(inplace=True)

df.drop(axis=1, columns="Id", inplace=True)

df.head()
qualiVars = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

df[qualiVars] = df[qualiVars].apply(preprocessing.LabelEncoder().fit_transform)
Xtrain = df.iloc[:,0:-1]

Ytrain = df.income

X_train, X_test, y_train, y_test = train_test_split(Xtrain,Ytrain,test_size=0.20)
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print("Floresta aleatória")

print('Acurácia: ',accuracy_score(y_test, rfc_pred))
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
log_pred = logmodel.predict(X_test)



print("Regressão logística")

print('Acurácia: ',accuracy_score(y_test, log_pred))
from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train,y_train)
svm_pred = svm.predict(X_test)

print("SVM")

print('Acurácia: ',accuracy_score(y_test, svm_pred))