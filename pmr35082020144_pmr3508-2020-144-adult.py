import pandas as pd

import numpy as np

import sklearn
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        index_col=['Id'], na_values="?")
adult.head()
adult.shape
adult["hours.per.week"].value_counts()
import matplotlib.pyplot as plt
adult["age"].value_counts().plot(kind="bar")
adult["workclass"].value_counts().plot(kind="bar")
adult["education.num"].value_counts().plot(kind="bar")
adult["marital.status"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
adult["race"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
adult_lapidado = adult.dropna()
adult_lapidado.shape
teste_adult = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        index_col=['Id'], na_values="?")
teste_adult.shape
teste_adult_lapidado = teste_adult.dropna()
teste_adult_lapidado.head()
Xadult = adult_lapidado[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = adult_lapidado[["income"]]

np.ravel(Yadult)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores
scores.mean()
knn = KNeighborsClassifier(n_neighbors=10)
scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores
scores.mean()
knn = KNeighborsClassifier(n_neighbors=15)
scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores
scores.mean()
knn = KNeighborsClassifier(n_neighbors=20)
scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores
scores.mean()
knn = KNeighborsClassifier(n_neighbors=25)
scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores
scores.mean()
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores

scores.mean()
Xteste = teste_adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
classificador = KNeighborsClassifier(n_neighbors=16)

classificador.fit(Xadult, Yadult)



predicao = classificador.predict(Xteste)

predicao
submissao = pd.DataFrame()
submissao[0] = Xteste.index

submissao[1] = predicao

submissao.columns = ["Id", "Income"]

submissao.head()
submissao.to_csv('submissao.csv',index = False)
import seaborn as sns
adult_lapidado["income"] = adult_lapidado["income"].map({"<=50K": 0, ">50K":1})

adult_lapidado["sex.num"] = adult_lapidado["sex"].map({"Male": 1, "Female":0})
sns.catplot(y="sex", x="income", kind="bar", data=adult_lapidado);