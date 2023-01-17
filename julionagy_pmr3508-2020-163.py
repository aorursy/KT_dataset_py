##importando bibliotecas



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import sklearn

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

##importando dados



train = pd.read_csv('../input/adult-pmr3508/train_data.csv', na_values = '?')

test = pd.read_csv('../input/adult-pmr3508/test_data.csv', na_values = '?')

sample = pd.read_csv('../input/adult-pmr3508/sample_submission.csv', na_values = '?')

extra = pd.read_fwf('../input/adult-pmr3508/Extra-file-from-UCI.txt', na_values = '?')




def treino():

    return train["education"].value_counts().plot(kind="bar")



treino()

def porcen(column):

    return column*100//float(column[-1])





TXA = pd.crosstab(train["age"],train["income"],margins=True)

TXA.apply(porcen, axis=1).plot()
educ = train.groupby(['education', 'income']).size().unstack()

educ['sum'] = train.groupby('education').size()

educ = educ.sort_values('sum', ascending = False)[['<=50K', '>50K']]

educ.plot(kind = 'bar', stacked = True)

train['native.country'].value_counts()
taredu = pd.crosstab(train['sex'],train['income'], margins= True)

taredu.apply(porcen, axis = 1).plot()
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
train["sex"] = train["sex"].apply(num_func)

train["race"] = train["race"].apply(num_func)

test["sex"] = test["sex"].apply(num_func)

test["race"] = test["race"].apply(num_func)
ntrain = train.dropna()
ntest = test

Xtrain = ntrain[["age","education.num","sex", "race", "capital.gain", "capital.loss", "hours.per.week"]]
Ytrain = ntrain.income
Xtest = ntest[["age","education.num","sex", "race", "capital.gain", "capital.loss", "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=24, p=1)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=15)
scores
knn.fit(Xtrain,Ytrain)
YtestPred = knn.predict(Xtest)
accuracy = np.mean(scores)

accuracy
id_index = pd.DataFrame({'Id' : list(range(len(YtestPred)))})

income = pd.DataFrame({'income' : YtestPred})

result = income

result
result.to_csv("submission.csv", index = True, index_label = 'Id')