import pandas as pd

import sklearn

import numpy as np
adult = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],na_values = "?")

testAdult = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv', 

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"], na_values = "?")
adult.head()
nadult = adult.dropna() #remove missing values

nTestAdult = testAdult.dropna()
testAdult.shape
#retirando a primeira linha com os titulos antigos

adult.drop(adult.index[0],inplace=True)

testAdult.drop(testAdult.index[0],inplace=True)
adult["Country"].value_counts()
import matplotlib.pyplot as plt #importando a biblioteca utilizada para plotar os graficos
adult["Age"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
from sklearn import preprocessing #importando a biblioteca para preprocessamento dos dados
numAdult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss",

                 "Hours per week"]].apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult.iloc[:,0:14]
Yadult = nadult.Target
numTestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss",

                         "Hours per week"]].apply(preprocessing.LabelEncoder().fit_transform)
XtestAdult = numTestAdult.iloc[:,0:14]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=3) #utilizando k = 3 neighbors
scores = cross_val_score(knn, Xadult, Yadult, cv=10) #calculando a taxa de erro por validação cruzada
scores
np.mean(scores)
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult.iloc[:,0:14]
Yadult = numAdult.Target
XtestAdult = numTestAdult.iloc[:,0:14]
knn = KNeighborsClassifier(n_neighbors=30)

#conjunto de atributos que sera utilizado

Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]

XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]
i = 5

best = 0

k = 0

while i<=30: #variando i de 5 a 30 para achar o numero de neighbors que permita a melhor estimativa

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, Xadult, Yadult, cv=10)

    mean = np.mean(scores)

    if mean > best:

        best = mean

        k = i

    i+=1
k # valor de k neighbors que permitiu a melhor taxa de erro por validação cruzada
best #melhor taxa de erro encontrada por validação cruzada
knn = KNeighborsClassifier(n_neighbors=k) #utilizando o knn para o k que possui a melhor taxa de erro calculada por validação cruzada
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
pred = []; #lista utilizada para passar os dados de YtestPred, que estão escritos em 0 ou 1, para <=50K e >50K

for i in range(len(YtestPred)-1):

    pred.append(0)

    if YtestPred[i] == 0:

        pred[i] = "<=50K"

    elif YtestPred[i] == 1:

        pred[i] = ">50K"

pred
#criar arquivo de resultado

savepath = "predictions.csv"

prev = pd.DataFrame(pred, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev