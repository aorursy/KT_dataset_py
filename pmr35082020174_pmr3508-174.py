import pandas as pd

import sklearn
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv", sep=',', na_values = '?')
adult.shape
adult.head()
adult["Country"].value_counts()
adult["Age"].value_counts()
import matplotlib.pyplot as plt
adult["Sex"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()


testAdult = pd.read_csv("../input/adult-pmr3508/test_data.csv", sep=',', na_values = '?')



nTestAdult = testAdult.dropna()

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
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
from sklearn import preprocessing
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult.iloc[:,0:14]
Yadult = numAdult.Target
XtestAdult = numTestAdult.iloc[:,0:14]
YtestAdult = numTestAdult.Target
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
Xadult = numAdult[["Age", "Workclass", "Education-Num", 

        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", 

        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)