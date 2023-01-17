import pandas as pd
import sklearn
adult = pd.read_csv("../input/dataset/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.head()
adult.shape
adult.describe()
nadult = adult.dropna()
nadult.shape
testAdult = pd.read_csv("../input/dataset/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testAdult.head()
testAdult.shape
ntestAdult = testAdult.dropna()
ntestAdult.shape
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult.income
XtestAdult = ntestAdult[["age","education.num", "capital.gain", "capital.loss", "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors = 3)
scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores
knn = KNeighborsClassifier(n_neighbors = 20)
scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores
from sklearn import preprocessing
import numpy as np
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
testNumAdult = ntestAdult.apply(preprocessing.LabelEncoder().fit_transform)
numAdult.head()
XnumAdult = numAdult[["workclass","education.num","marital.status" ,"capital.gain","capital.loss", "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors = 33)
scores = cross_val_score(knn, XnumAdult, Yadult, cv = 10)
scores
np.average(scores)
XtestNumAdult = testNumAdult[["workclass","education.num","marital.status" ,"capital.gain","capital.loss", "hours.per.week"]]
knn.fit(XnumAdult, Yadult)
YtestPred = knn.predict(XtestNumAdult)
evaluation = pd.DataFrame(ntestAdult.Id)
evaluation["income"] = YtestPred
evaluation
evaluation.to_csv("evaluation.csv", index=False)