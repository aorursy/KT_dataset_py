import pandas as pd
import sklearn
import matplotlib.pyplot as plt
adult = pd.read_csv("../input/adultb/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
adult.head()
adult["native.country"].value_counts()
adult["race"].value_counts()
adult["workclass"].value_counts()
adult["education"].value_counts()
adult["race"].value_counts().plot(kind="pie")
nadult = adult.dropna()
adult.shape
nadult.shape
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
testAdult = pd.read_csv("../input/adultb/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
nTestAdult = testAdult

Xadult = nadult[["age", "education.num","hours.per.week","capital.gain", "capital.loss"]]
Yadult = nadult.income

XtestAdult = nTestAdult[["age", "education.num","hours.per.week","capital.gain", "capital.loss"]]
maiork = 1
maiormedia = 0

for k in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    if np.mean(scores) > maiormedia:
        maiormedia = np.mean(scores)
        maiork = k
    
maiork
knn = KNeighborsClassifier(n_neighbors = maiork)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
np.mean(scores)
knn.fit(Xadult, Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
result = np.vstack((nTestAdult["Id"], YtestPred)).T
x = ["id","income"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("results.csv", index = False)
Resultado