import pandas as pd

import sklearn

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing
trainAdult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Income"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
trainAdult.shape
trainAdult.head()
trainAdult.drop(trainAdult.index[0],inplace=True) #tirando primeira linha

ntrainAdult = trainAdult.dropna() #retirando linhas com dados faltantes.
ntrainAdult
ntrainAdult.shape
ntrainAdult["Age"].value_counts().plot(kind="bar")
ntrainAdult["Sex"].value_counts().plot(kind="bar")
#ntrainAdult["Race"].value_counts().plot(kind="bar")
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
testAdult.drop(testAdult.index[0],inplace=True) #tirando primeira linha

ntestAdult = testAdult

ntestAdult.fillna(method ='ffill', inplace = True) #preenchendo linhas
ntestAdult
ntestAdult['Workclass'].value_counts().plot(kind="bar") 

#ntestAdult['Age'].value_counts().plot(kind="bar") 
XtrainAdult = ntrainAdult[["Age", "Workclass", "Education-Num", "Marital Status", "Relationship", "Capital Gain", "Capital Loss", "Hours per week", "Country"]].apply(preprocessing.LabelEncoder().fit_transform)

YtrainAdult = ntrainAdult.Income 



YtrainAdult.shape
YtrainAdult.head()
XtestAdult = ntestAdult[["Age", "Workclass", "Education-Num", "Marital Status", "Relationship", "Capital Gain", "Capital Loss", "Hours per week", "Country"]].apply(preprocessing.LabelEncoder().fit_transform) 

YtestAdult = ntestAdult.Income 
##procurando o melhor valor para n_neighbors



#n_n = 20

#cval = 10

#acuracia_esperada = 0

#acuracia_best = 0

#array_acuracia = []

#while n_n <=50:

#    knn = KNeighborsClassifier(n_neighbors= n_n)

#    knn.fit(XtrainAdult,YtrainAdult)

#    scores = cross_val_score(knn, XtrainAdult, YtrainAdult, cv=cval)

#    YtestPred = knn.predict(XtestAdult)

#    total = 0

#    for i in scores:

#        total += i

#    acuracia_esperada = total/cval

#    array_acuracia.append(acuracia_esperada)

#    if acuracia_esperada > acuracia_best:

#        acuracia_best = acuracia_esperada

#    n_n = n_n + 1

#acuracia_best
# array_acuracia

## o melhor valor foi encontrado para n_n = 31
knn = KNeighborsClassifier(n_neighbors=31)

knn.fit(XtrainAdult,YtrainAdult)
cval = 10

scores = cross_val_score(knn, XtrainAdult, YtrainAdult, cv=cval)

scores
YtestPred = knn.predict(XtestAdult)

YtestPred
total = 0

for i in scores:

    total += i

acuracia_esperada = total/cval

acuracia_esperada
YtestPred

YtestPred.shape
# Preparando arquivo para submissao

savepath = "predictions5.csv"

prev = pd.DataFrame(YtestPred, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev