import pandas as pd

import sklearn

import matplotlib.pyplot as plt
train_adult = pd.read_csv("../Machlearn/adult.data.txt" ,names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

test_adult =  pd.read_csv("../Machlearn/adult.test.txt" ,names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
train_adult.shape
test_adult.shape
train_adult.head()
train_adult["Country"].value_counts()
train_adult["Age"].value_counts().plot(kind="bar")
train_adult["Sex"].value_counts().plot(kind="bar")
train_adult["Education"].value_counts().plot(kind="bar")
train_adult["Occupation"].value_counts().plot(kind="bar")
nadult = train_adult.dropna()

nadult
ntest_adult = test_adult.dropna()
ntest_adult
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Yadult = nadult.Target
XtestAdult = ntest_adult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

YtestAdult = ntest_adult.Target
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
YtestPred = knn.predict(XtestAdult)

YtestPred
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = ntest_adult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult.iloc[:,0:14]

XtestAdult = numTestAdult.iloc[:,0:14]
Yadult = numAdult.Target

YtestAdult = numTestAdult.Target
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
Xadult = numAdult[["Age", "Workclass", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)