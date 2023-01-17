# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
adult = pd.read_csv("../input/train_data.csv")
#retirando os dados faltantes:
nadult = adult.dropna()
nadult.head()
nadult["age"].value_counts().plot(kind="bar")
nadult[nadult['income']=="<=50K"].age.value_counts()
nadult[nadult['income']==">50K"].age.value_counts()
nadult[nadult['income']==">50K"].age.value_counts().plot(kind="bar")
nadult["sex"].value_counts().plot(kind="bar")
nadult[nadult['income']==">50K"].sex.value_counts().plot(kind="bar")
nadult[nadult['income']=="<=50K"].sex.value_counts().plot(kind="bar") #a proporção não é mantida
testAdult = pd.read_csv("../input/test_data.csv")
nTestAdult = testAdult.dropna()
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
numAdult.rename(columns={"hours.per.week":"hours_per_week", "native.country":"native_contry", "capital.gain":"capital_gain", "capital.loss":"capital_loss"},inplace=True)
numAdult
numTestAdult.rename(columns={"hours.per.week":"hours_per_week", "native.country":"native_contry", "capital.gain":"capital_gain", "capital.loss":"capital_loss"},inplace=True)
#Primeiro teste: seleção de atributos numéricos, com kNN para k=3. 
Xadult = numAdult[["age", "fnlwgt", "education", "occupation", "capital_gain", "sex", "hours_per_week", "native_contry"]]
Yadult = numAdult.income
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)  #validação cruzada
scores
knn.fit(Xadult,Yadult)
#Para K=30
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
