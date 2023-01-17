import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt
bd = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
bd.shape

bd.head()

bd["native.country"].value_counts()

bd["age"].value_counts().plot(kind="bar")

bd["sex"].value_counts().plot(kind="pie")

bd["relationship"].value_counts().plot(kind="bar")
bd["education"].value_counts().plot(kind="bar")

bd["occupation"].value_counts().plot(kind="bar")

bd["workclass"].value_counts().plot(kind = "bar")
bd["hours.per.week"].value_counts().plot(kind = "bar")
nbd = bd.dropna()

nbd.shape

testbd = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
Xbd = nbd[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

Ybd = nbd.income

Xtestbd = testbd[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, Xbd, Ybd, cv=10)

scores

knn.fit(Xbd,Ybd)

YtestPred = knn.predict(Xtestbd)

YtestPred

accuracy = np.mean(scores)

accuracy

knn = KNeighborsClassifier(n_neighbors=28)

knn.fit(Xbd,Ybd)

scores = cross_val_score(knn, Xbd, Ybd, cv=10)

scores

YtestPred = knn.predict(Xtestbd)

accuracy = np.mean(scores)

accuracy
id_index = pd.DataFrame({'Id' : list(range(len(YtestPred)))})

income = pd.DataFrame({'income' : YtestPred})

result = income

result
result.to_csv("submissionBorb.csv", index = True, index_label = 'Id')