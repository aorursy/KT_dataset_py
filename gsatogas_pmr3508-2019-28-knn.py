#importar as blibliotecas que vamos utilizar

import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

from sklearn import preprocessing
adult = pd.read_csv('../input/adult-pmr3508/train_data.csv',

        engine='python',

        sep=r'\s*,\s*',

        na_values='?')
adult.shape
adult.head()
adult["hours.per.week"].value_counts()
adult["age"].value_counts()
adult["education"].value_counts().plot(kind='bar')
adult["race"].value_counts().plot(kind='pie',autopct='%1.1f%%')
adult["sex"].value_counts().plot(kind='pie',autopct='%1.1f%%')
adult["marital.status"].value_counts().plot(kind='bar')
adult["occupation"].value_counts().plot(kind='bar')
#retirar os missing data da base de treino

clean_adult = adult.dropna()
clean_adult
TestAdult = pd.read_csv('../input/adult-pmr3508/test_data.csv',

        engine='python',

        sep=r'\s*,\s*',

        na_values='?')
#retirar os missing data da base de teste

clean_test = TestAdult.dropna()
clean_test.head()
main=["age","education.num", "capital.gain", "capital.loss","hours.per.week"]
Xtrain = clean_adult[main]

Ytrain = clean_adult.income

Xtest = clean_test[main]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score as css_val

from sklearn.metrics import accuracy_score as acc_sc
#Apos varios testes chegou a conclusao que 22 foi o melhor parametro

knn=KNeighborsClassifier(n_neighbors=22)
scores = css_val(knn, Xtrain, Ytrain, cv=10)

scores
knn.fit(Xtrain,Ytrain)
Ypred = knn.predict(Xtest)

Ypred
id_index = pd.DataFrame({'Id' : list(range(len(Ypred)))})

income = pd.DataFrame({'income' : Ypred})

predict = id_index.join(income)
predict.to_csv("submission.csv", index = False)