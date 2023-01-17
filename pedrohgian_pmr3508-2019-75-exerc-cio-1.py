import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline

import os

os.listdir('/kaggle/input/pmr3508-tarefa-1-3508-adult-dataset')
adult = pd.read_csv('../input/adult-dataset/adult.data',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adult.head()
adult.shape #Há 32561 atributos, e 15 parâmetros.
list(adult)
adult["Age"].value_counts().plot(kind="bar")

adult["Country"].value_counts().plot(kind="bar")

adult["Sex"].value_counts().plot(kind="pie")

adult["Martial Status"].value_counts().plot(kind="bar")

adult["Race"].value_counts().plot(kind="pie")
adult["Occupation"].value_counts().plot(kind="bar")

##adult["fnlwgt"].value_counts().plot(kind="bar")
adult.isnull().sum().plot(kind="bar")

nAdult = adult.dropna() 

nAdult

nAdult.shape #Há menos linhas de atributos!
testAdult = pd.read_csv('../input/adult-dataset/adult.test',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

testAdult.shape
nTestAdult = testAdult.dropna()

nTestAdult
nTestAdult.shape

#Consideraremos inicialmente todos os valores numéricos

Xadult = nAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Xadult.head()
Yadult = nAdult.Target

Yadult.head


#Consideraremos inicialmente todos os valores numéricos

XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

YtestAdult = nTestAdult.Target

#Aplicação da Validação Cruzada e Treinamento do modelo:

import sklearn

from sklearn.neighbors import KNeighborsClassifier



#Primeiro teste com k=3

knn = KNeighborsClassifier(n_neighbors=3)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores.mean()

knn = KNeighborsClassifier(n_neighbors=30)

scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores.mean()

from sklearn import preprocessing

numAdult = nAdult.apply(preprocessing.LabelEncoder().fit_transform)

numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)


Xadult = numAdult.iloc[:,0:14]
Yadult = numAdult.Target
XtestAdult = numTestAdult.iloc[:,0:14]
YtestAdult = numTestAdult.Target
knn = KNeighborsClassifier(n_neighbors=30)

scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores.mean()
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
from sklearn.metrics import accuracy_score



accuracy_score(YtestAdult,YtestPred)
Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
knn = KNeighborsClassifier(n_neighbors=30)

scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores.mean()
knn.fit(Xadult,Yadult)

YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)