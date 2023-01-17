import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import os
adult = pd.read_csv("../input/pmr3508-22018-1/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
adult = adult.dropna()
adult.shape
adult.head()
for x in ["Id","fnlwgt","education","native.country"]:
    adult.pop(x)
adult["sex"] = sklearn.preprocessing.LabelEncoder().fit_transform(adult["sex"])
adult.head()
encoder = ce.TargetEncoder(cols=["workclass","marital.status","occupation","relationship","race"])
adultpreprocess = adult.copy()
adultpreprocess["income"] = sklearn.preprocessing.LabelEncoder().fit_transform(adultpreprocess["income"])
encoder.fit(adultpreprocess, adultpreprocess["income"])
adultpreprocess.head()
adult.head()
adult = encoder.transform(adult, adult["income"])
adult.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
Xadult = adult[["age","education.num","sex","capital.gain",
           "capital.loss","hours.per.week","workclass",
           "marital.status","occupation","relationship","race"]]
Yadult = adult.income
list = []

i=1
knn = KNeighborsClassifier(n_neighbors=i)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=5)
accuracy = sum(scores)/len(scores)
list.append([i,accuracy])

i=5
while (i <= 70):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xadult,Yadult)
    scores = cross_val_score(knn, Xadult, Yadult, cv=5)
    accuracy = sum(scores)/len(scores)
    list.append([i,accuracy])
    i += 5
list
resultados = pd.DataFrame(list,columns=["k","accuracy"])
resultados.plot(x="k",y="accuracy",style="")
resultados.accuracy.max()
adultTest = pd.read_csv("../input/pmr3508-22018-1/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
#adultTest = adultTest.dropna()
adultTest
adultTest["income"] = ""
adultTestpreprocess = adultTest.copy()
for x in ["Id","fnlwgt","education","native.country"]:
    adultTestpreprocess.pop(x)
adultTestpreprocess["sex"] = sklearn.preprocessing.LabelEncoder().fit_transform(adultTestpreprocess["sex"])
adultTestpreprocess.head()
adultTestpreprocess = encoder.transform(adultTestpreprocess)
adultTestpreprocess.head()
XadultTest = adultTestpreprocess[["age","education.num","sex","capital.gain",
           "capital.loss","hours.per.week","workclass",
           "marital.status","occupation","relationship","race"]]
YadultTest = adultTestpreprocess.income

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(Xadult,Yadult)
YadultTest = knn.predict(XadultTest)
YadultTest
Evaluation = pd.DataFrame(adultTest.Id)
Evaluation["income"] = YadultTest
Evaluation
Evaluation.to_csv("Evaluation.csv", index=False)
