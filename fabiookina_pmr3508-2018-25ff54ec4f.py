# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/adult-data"))

# Any results you write to the current directory are saved as output.
adult = pd.read_csv("../input/adult-data/train_data.csv",
        na_values="?")
adult.shape
adult.head()
adult["native.country"].value_counts()
import matplotlib.pyplot as plt
adult["age"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()
nadult

testAdult = pd.read_csv("../input/adult-data/test_data.csv",
        na_values="?")


nTestAdult = testAdult.dropna()
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult["income"]
nTestAdult
XtestAdult = nTestAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
knn = KNeighborsClassifier(n_neighbors=50)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
knn = KNeighborsClassifier(n_neighbors=20)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
from sklearn import preprocessing

numAdult = nadult.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)

numTestAdult = nTestAdult.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)

Xadult=numAdult
Yadult = nadult["income"]
YtestAdult=numTestAdult
Xadult
Yadult
XtestAdult
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
Xadult = numAdult[["age", "workclass", "education.num", 
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
Xadult = numAdult[["age", "workclass", "education.num", 
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=40)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
Xadult = numAdult[["age", "workclass", "education.num", 
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=57)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()

Xadult = numAdult[["age", "workclass", "education.num", 
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
XtestAdult = numTestAdult[["age", "workclass", "education.num", 
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
nTestAdult["Id"]
YtestPred
YtestPred2=['']*16280
YtestPred
jj=0
kk=0
for ii in nTestAdult["Id"]:
    while jj < ii:
        YtestPred2[jj]="<=50K"
        jj=jj+1
    if ii == jj:
        YtestPred2[jj]=YtestPred[kk]
        jj=jj+1
        kk=kk+1
ID=list(range(0,16280))
result=np.column_stack((ID,YtestPred2))
x=["Id","income"]
Prediction = pd.DataFrame(columns = x, data = result)
Prediction.to_csv("results.csv", index = False)
Prediction