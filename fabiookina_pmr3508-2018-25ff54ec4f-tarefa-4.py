# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
adult = pd.read_csv("../input/train_data.csv",
        na_values="?")
adult.shape
adult.head()
adult["native.country"].value_counts()
adult["age"].value_counts().plot(kind="bar")
adult["age"].mean()
adult["sex"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()
testAdult = pd.read_csv("../input/test_data.csv",
        na_values="?")


nTestAdult = testAdult.dropna()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


from sklearn import preprocessing



numAdult = nadult.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)

numTestAdult = nTestAdult.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)

Xadult=numAdult
Yadult = nadult["income"]


Xadult = numAdult[["age", "workclass", "education.num", 
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
XtestAdult = numTestAdult[["age", "workclass", "education.num", 
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=57)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)

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
Prediction.to_csv("results_knn.csv", index = False)
Prediction


from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(numAdult, Yadult)
cross_val_score(MNB,Xadult,Yadult,cv=10).mean()
MNB.fit(Xadult,Yadult)
YtestPred = MNB.predict(XtestAdult)

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
Prediction.to_csv("results_MNB.csv", index = False)
Prediction

from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(Xadult, Yadult)
cross_val_score(GNB, Xadult, Yadult, cv=10).mean()
GNB.fit(Xadult,Yadult)
YtestPred = GNB.predict(XtestAdult)

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
Prediction.to_csv("results_GNB.csv", index = False)
Prediction

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state=0)
cross_val_score(DTC, Xadult, Yadult, cv=10).mean()
DTC.fit(Xadult,Yadult)
YtestPred = DTC.predict(XtestAdult)

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
Prediction.to_csv("results_DTC.csv", index = False)
Prediction

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=300, random_state=0)
cross_val_score(RFC, Xadult, Yadult, cv=10).mean()
RFC.fit(Xadult,Yadult)
YtestPred = RFC.predict(XtestAdult)

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
Prediction.to_csv("results_RFC.csv", index = False)
Prediction

from sklearn.neural_network import MLPClassifier
MLPC = MLPClassifier(solver='lbfgs', alpha=1000, hidden_layer_sizes=(100, 5), random_state=1)
cross_val_score(MLPC, Xadult, Yadult, cv=10).mean()
MLPC.fit(Xadult,Yadult)
YtestPred = MLPC.predict(XtestAdult)

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
Prediction.to_csv("results_MLPC.csv", index = False)
Prediction
