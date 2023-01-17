import sklearn
import pandas as pd
import matplotlib.pyplot as plt
traindata = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=',',

        engine='python',

        na_values="?")
traindata.shape
traindata.describe()
testdata = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testdata.shape
traindata.head()
testdata.head()

traindata["native.country"].value_counts()

traindata["age"].value_counts().plot(kind="bar")
traindata["sex"].value_counts().plot(kind="pie")

traindata["education"].value_counts().plot(kind="bar")

traindata["occupation"].value_counts().plot(kind="bar")

def percent(colum):

    return colum*100//float(colum[-1])

targetxage = pd.crosstab(traindata["age"],traindata["income"],margins=True)

targetxage.apply(percent,axis=1).plot()


targetxeducationnum = pd.crosstab(traindata["education.num"],traindata["income"],margins=True)

targetxeducationnum.apply(percent,axis=1).plot()
targetxeducationnum = pd.crosstab(traindata["income"],traindata["sex"],margins=True)

targetxeducationnum.apply(percent,axis=0)
targetxeducationnum = pd.crosstab(traindata["sex"],traindata["income"],margins=True)

targetxeducationnum.apply(percent,axis=1).plot()
#Aqui vamos transformar os labels Sex e Races que são strings em floats, a partir da associação com sua porcentagem de aparição

no_number = ["Male","Female"]

number = ["3","1"]

races=["Asian-Pac-Islander","White","Black","Amer-Indian-Eskimo","Other"]

perc=["26", "25", "12", "11","9"]

no_number += races

number += perc

def num_func(label):

    for i in range(len(number)):

        if label == no_number[i]:

            return number[i]

    return label
#A partir da função .apply da Pandas, podemos aplicar ao CSV a nossa alteração numérica

traindata["sex"] = traindata["sex"].apply(num_func)

traindata["race"] = traindata["race"].apply(num_func)

testdata["sex"] = testdata["sex"].apply(num_func)

testdata["race"] = testdata["race"].apply(num_func)

ntraindata = traindata.dropna()

ntraindata
ntestdata = testdata

ntestdata.shape
Xtrain = ntraindata[["age","education.num","sex", "race", "capital.gain", "capital.loss", "hours.per.week"]]

Ytrain = ntraindata.income

Xtest = ntestdata[["age","education.num","sex", "race", "capital.gain", "capital.loss", "hours.per.week"]]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=24, p=1)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, Xtrain, Ytrain, cv=15)

scores

knn.fit(Xtrain,Ytrain)

YtestPred = knn.predict(Xtest)



import numpy as np
accuracy = np.mean(scores)

accuracy
id_index = pd.DataFrame({'Id' : list(range(len(YtestPred)))})

income = pd.DataFrame({'income' : YtestPred})

result = income

result
result.to_csv("submission.csv", index = True, index_label = 'Id')
