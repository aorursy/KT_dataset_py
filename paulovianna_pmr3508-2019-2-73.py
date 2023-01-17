import pandas as pd

import sklearn
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.shape
adult.head()
adult["Country"].value_counts()
import matplotlib.pyplot as plt
adult["Age"].value_counts().plot(kind="bar")
adult["Sex"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()
nadult
nadult.drop('Id',axis=0,inplace=True)

nadult
testAdult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
nTestAdult = testAdult.dropna()
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult = nadult.Target
XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
YtestAdult = nTestAdult.Target
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators = 10)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RFC, Xadult, Yadult, cv=10)
scores
RFC.fit(Xadult,Yadult)

XtestAdult
XtestAdult.drop('Id',axis=0,inplace=True)

XtestAdult
YtestAdult.drop('Id',axis=0,inplace=True)

YtestAdult
YtestPred = RFC.predict(XtestAdult)
YtestPred
from sklearn.metrics import accuracy_score
accuracy_score(YtestAdult,YtestPred)
import pandas as pd

import sklearn
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.shape
adult.head()
adult["Country"].value_counts()
import matplotlib.pyplot as plt
adult["Age"].value_counts().plot(kind="bar")
adult["Sex"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()
nadult
nadult.drop('Id',axis=0,inplace=True)

nadult
testAdult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
nTestAdult = testAdult.dropna()
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult = nadult.Target
XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
YtestAdult = nTestAdult.Target
from sklearn.naive_bayes import BernoulliNB
BernNB = BernoulliNB(binarize = True)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(BernNB, Xadult, Yadult, cv=10)
scores
BernNB.fit(Xadult,Yadult)

XtestAdult
XtestAdult.drop('Id',axis=0,inplace=True)

XtestAdult
YtestAdult.drop('Id',axis=0,inplace=True)

YtestAdult
YtestPred = BernNB.predict(XtestAdult)
YtestPred
from sklearn.metrics import accuracy_score
accuracy_score(YtestAdult,YtestPred)
import pandas as pd

import sklearn
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.shape
adult.head()
adult["Country"].value_counts()
import matplotlib.pyplot as plt
adult["Age"].value_counts().plot(kind="bar")
adult["Sex"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()
nadult
nadult.drop('Id',axis=0,inplace=True)

nadult
testAdult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
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

XtestAdult
XtestAdult.drop('Id',axis=0,inplace=True)

XtestAdult
YtestAdult.drop('Id',axis=0,inplace=True)

YtestAdult
YtestPred = knn.predict(XtestAdult)
YtestPred
from sklearn.metrics import accuracy_score
accuracy_score(YtestAdult,YtestPred)