import pandas as pd

import sklearn

import numpy as np
train="../input/adult-pmr3508/train_data.csv"

test="../input/adult-pmr3508/test_data.csv"
test = pd.read_csv(test,

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
train = pd.read_csv(train,

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country","Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

train.drop(train.index[0],inplace=True)

test.drop(test.index[0],inplace=True)
test.head() 
train.head() 
train["Country"].value_counts()
import matplotlib.pyplot as plt
train["Age"].value_counts().plot(kind="bar")
train["Sex"].value_counts().plot(kind="bar")
train["Education"].value_counts().plot(kind="bar")
train["Occupation"].value_counts().plot(kind="bar")
ntrain = train.dropna()
train.shape
#fazendo tudo igual para o test
ntest = test.dropna()
test.shape
Xtrain = ntrain[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Ytrain = ntrain.Target
Xtest = ntest[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=3)

scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)

knn.fit(Xtrain,Ytrain)

YtestPred = knn.predict(Xtest) 

YtestPred
best_mean = 0

best_k = 0

for k in range(3,30):

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)

    mean = np.mean(scores)

    if (mean > best_mean):

        best_mean = mean

        best_k = k

        

    

    
knn = KNeighborsClassifier(n_neighbors=k)

scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)

knn.fit(Xtrain,Ytrain)

YtestPred = knn.predict(Xtest) 

YtestPred
savepath = "YtestPred.csv"

prev = pd.DataFrame(YtestPred, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev