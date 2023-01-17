# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
adult = pd.read_csv('../input/adult-income-dataset/adult.data',names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adult.shape
adult.head()
adult["Country"].value_counts()
adult["Age"].value_counts().plot(kind="bar")
adult["Sex"].value_counts().plot(kind="bar")

adult["Education"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
nadult= adult.dropna()

nadult
test_adult = pd.read_csv('../input/adult-income-dataset/adult.test',names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

ntest_adult = test_adult.dropna()
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Yadult = nadult.Target
Xtest_adult = ntest_adult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Ytest_adult = ntest_adult.Target

Ytest_adult = Ytest_adult.values



for i in range (len(Ytest_adult)):

    Ytest_adult[i] = Ytest_adult[i][:-1]

knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores

YtestPred = knn.predict(Xtest_adult)
YtestPred
accuracy_score(Ytest_adult,YtestPred)