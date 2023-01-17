import warnings



import pandas as pd

import numpy as np



import sklearn

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing



%matplotlib inline

plt.style.use('seaborn')



trainAdult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Income"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



trainAdult.info()



trainAdult.shape



trainAdult.head()



trainAdult.select_dtypes('object').head()



trainAdult.drop(trainAdult.index[0],inplace=True)

ntrainAdult = trainAdult.dropna()



ntrainAdult.shape



ntrainAdult["Age"].value_counts().plot(kind="bar")



testAdult = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Income"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



testAdult.head()



testAdult.shape



testAdult.drop(testAdult.index[0],inplace=True)

ntestAdult = testAdult

ntestAdult.fillna(method ='ffill', inplace = True)



ntestAdult['Education'].value_counts().plot(kind="bar") 



ColtrainAdult = ntrainAdult[["Age", "Workclass", "Education-Num", "Marital Status", "Relationship", "Capital Gain", "Capital Loss", "Hours per week", "Country"]].apply(preprocessing.LabelEncoder().fit_transform)

RowtrainAdult = ntrainAdult.Income 



RowtrainAdult.shape



RowtrainAdult.head()



ColtestAdult = ntestAdult[["Age", "Workclass", "Education-Num", "Marital Status", "Relationship", "Capital Gain", "Capital Loss", "Hours per week", "Country"]].apply(preprocessing.LabelEncoder().fit_transform) 

RowtestAdult = ntestAdult.Income 



knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(ColtrainAdult,RowtrainAdult)



cval = 4

scores = cross_val_score(knn, ColtrainAdult, RowtrainAdult, cv=cval)

scores



RowtestPred = knn.predict(ColtestAdult)

RowtestPred



total = 0

for i in scores:

    total += i

acuracia_esperada = total/cval

acuracia_esperada



RowtestPred

RowtestPred.shape



savepath = "predictions2.csv"

prev = pd.DataFrame(RowtestPred, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev












