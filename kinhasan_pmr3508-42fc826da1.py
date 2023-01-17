import os
print(os.listdir("../input/"))
import pandas as pd
import sklearn as sk
import matplotlib as mat
test = pd.read_csv("../input/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train = pd.read_csv("../input/train_data.csv")
train.shape
train.head()
train["native.country"].value_counts()
train["race"].value_counts()
train["occupation"].value_counts()
train["sex"].value_counts().plot(kind="bar")
ntrain=train.dropna()
test
ntrain
ntest=test.dropna()
Xtrain=ntrain[["age","hours.per.week","capital.gain","capital.loss"]]
Ytrain=ntrain.income
Xtest=ntest[["age","hours.per.week","capital.gain","capital.loss"]]

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
prediction = pd.DataFrame(index = ntest.Id)
prediction
prediction['income'] = YtestPred
prediction