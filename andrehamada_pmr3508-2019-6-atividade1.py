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
import sklearn
adult = pd.read_csv("/kaggle/input/pmr35086/adult_data.txt",

                    names=[

                    "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.head()
nAdult = adult.dropna()
testAdult = pd.read_csv("/kaggle/input/pmr35086/adult_test.txt",

                    names=[

                    "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testAdult
testAdult.drop([0], axis = 0)
nTestAdult = testAdult.dropna()
nTestAdult.shape
xAdult = nAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
yAdult = nAdult.Target
xTestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
yTestAdult = nTestAdult["Target"]
yTestAdult.shape
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, xAdult, yAdult, cv=10)
scores
knn.fit(xAdult,yAdult)
yTestPred = knn.predict(xTestAdult)
yTestPred
yTestAdult
from sklearn.metrics import accuracy_score
print(accuracy_score(yTestPred, yTestAdult)*100)
accuracy_score(yTestPred, yTestAdult)
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(xAdult,yAdult)
scores = cross_val_score(knn, xAdult, yAdult, cv=10)
yTestPred = knn.predict(xTestAdult)
accuracy_score(yTestAdult,yTestPred)
from sklearn import preprocessing
numAdult = nAdult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
xAdult = numAdult[["Age",  "Education-Num",  "Race", "Sex", "Capital Gain", "Capital Loss",]]
xTestAdult = numTestAdult[["Age",  "Education-Num",  "Race", "Sex", "Capital Gain", "Capital Loss",]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(xAdult,yAdult)
scores = cross_val_score(knn, xAdult, yAdult, cv=10)
yTestPred = knn.predict(xTestAdult)
accuracy_score(yTestAdult,yTestPred)