# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import sklearn
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        skiprows=1)
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
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        skiprows=1)
testSampleAdult = pd.read_csv("/kaggle/input/adult-pmr3508/sample_submission.csv",

        names=["Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        skiprows=1)
testAdult = pd.concat([testAdult, testSampleAdult], axis=1, join='inner').sort_index()
nTestAdult = testAdult.dropna()
nTestAdult
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
Yadult
accuracy_score(YtestAdult,YtestPred)
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