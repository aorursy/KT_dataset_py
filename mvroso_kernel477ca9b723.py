import pandas as pd

import sklearn
adult = pd.read_csv("/Users/mvros/Documents/POLI/2019 - 2º Semestre/PMR3508 - Aprendizado de Máquina e Reconhecimento de Padrões/adult.data.txt",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.shape
adult.head()
nadult = adult.dropna()
nadult
testAdult = pd.read_csv("/Users/mvros/Documents/POLI/2019 - 2º Semestre/PMR3508 - Aprendizado de Máquina e Reconhecimento de Padrões/adult.test.txt",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
nTestAdult = testAdult.dropna()
import matplotlib.pyplot as plt

nadult["Race"].value_counts().plot(kind="bar")
nadult["Workclass"].value_counts().plot(kind="bar")
nadult["Relationship"].value_counts().plot(kind="bar")
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Yadult = nadult.Target
XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

YtestAdult = nTestAdult.Target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
from sklearn.metrics import accuracy_score
accuracy_score(YtestAdult,YtestPred)
from sklearn import preprocessing
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult.iloc[:,0:14]
Yadult = numAdult.Target
XtestAdult = numTestAdult.iloc[:,0:14]
YtestAdult = numTestAdult.Target
knn = KNeighborsClassifier(n_neighbors=27)

knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)

accuracy_score(YtestAdult,YtestPred)
Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex","Hours per week"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex","Hours per week"]]
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)