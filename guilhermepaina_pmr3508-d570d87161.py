import os
print(os.listdir("../input"))
import pandas as pd
import sklearn as sk
test = pd.read_csv("../input/data-tarefa1/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train = pd.read_csv("../input/data-tarefa1/train_data.csv")
train.shape
train.head()
train["native.country"].value_counts()
import matplotlib.pyplot as plt
train["age"].value_counts().plot(kind="bar")
train["sex"].value_counts().plot(kind="bar")
train["education"].value_counts().plot(kind="bar")
train["occupation"].value_counts().plot(kind="bar")
train["race"].value_counts().plot(kind="bar")
Ntrain = train.dropna()
Ntrain
test
Ntest = test.dropna()
Ntest
Xtrain = Ntrain[["age","hours.per.week","capital.gain","capital.loss"]]
Ytrain = Ntrain.income
Xtest= Ntest[["age","hours.per.week","capital.gain","capital.loss"]]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
knn.fit(Xtrain, Ytrain)
scores
YtestPred = knn.predict(Xtest)
YtestPred
knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(Xtrain,Ytrain)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
prediction = pd.DataFrame(index = Ntest.Id)
prediction['income'] = YtestPred
prediction