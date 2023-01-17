import pandas as pd

import sklearn

import matplotlib.pyplot as plt
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.shape
adult
list(adult)
adult.describe()
adult["Country"].value_counts()
us = adult[adult["Country"] == "United-States"]

mex = adult[adult["Country"] == "Mexico"]

phi = adult[adult["Country"] == "Philippines"]

de = adult[adult["Country"] == "Germany"]

can = adult[adult["Country"] == "Canada"]

pur = adult[adult["Country"] == "Puerto-Rico"]
efeito = {us[us["Target"] == ">50K"].shape[0] / us.shape[0],

          mex[mex["Target"] == ">50K"].shape[0] / mex.shape[0],

          phi[phi["Target"] == ">50K"].shape[0] / phi.shape[0],

          de[de["Target"] == ">50K"].shape[0] / de.shape[0],

          can[can["Target"] == ">50K"].shape[0] / can.shape[0],

          pur[pur["Target"] == ">50K"].shape[0] / pur.shape[0]}        

efeito
adult["Target"].value_counts().plot(kind = "bar")
adult["Sex"].value_counts().plot(kind = "bar")
adult["Age"][adult["Target"] == ">50K"].value_counts().plot(kind="bar")
adult.shape
adult =  adult.dropna()

adult.shape
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testAdult.shape
testAdult = testAdult.dropna()

testAdult.shape
testAdult.head()
from sklearn import preprocessing
numAdult = adult.apply(preprocessing.LabelEncoder().fit_transform)
numAdult.head()
numTestAdult = testAdult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult.head()
attributes = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]
adultX =  numAdult[attributes].apply(preprocessing.LabelEncoder().fit_transform)

adultY =  numAdult.Target
testAdultX = numTestAdult[attributes].apply(preprocessing.LabelEncoder().fit_transform)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
attributes = ["Age", "Education-Num", "Martial Status","Relationship", "Race", "Sex", "Capital Gain", "Capital Loss"]
adultX =  numAdult[attributes].apply(preprocessing.LabelEncoder().fit_transform)

adultY =  numAdult.Target
testAdultX = numTestAdult[attributes].apply(preprocessing.LabelEncoder().fit_transform)
knn = KNeighborsClassifier(n_neighbors=60)

scores = cross_val_score(knn, adultX, adultY, cv=10)

scores
knn.fit(adultX, adultY)

testPred = knn.predict(testAdultX)

testPred
income = pd.DataFrame(testPred)

income.to_csv("submission.csv",header = ["income"], index_label = "Id")