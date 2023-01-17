# bibliotecas

import numpy as np
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt

#sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
# DADOS DE TREINO
AdultTrain = pd.read_csv("../input/adultba/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

# DADOS DE TESTE
AdultTest = pd.read_csv('../input/adultba/test_data.csv',
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
# nadult - tira linhas com dados faltantes
nAdultTrain = AdultTrain.dropna()
nAdultTest = AdultTest.dropna()
# num linhas, num colunas
a = AdultTrain.shape
b = nAdultTrain.shape
c = AdultTest.shape
d = nAdultTest.shape
print ("dados de treino:",a,b,", dados de teste:",c,d)
# formato da tabela
AdultTrain.head()
AdultTrain["native.country"].value_counts()
AdultTrain["native.country"].value_counts().plot(kind = "bar")
AdultTrain["age"].value_counts().plot(kind = "bar")
AdultTrain["marital.status"].value_counts().plot(kind = "bar")
AdultTrain["sex"].value_counts().plot(kind = "pie")
Xadult = nAdultTrain[["age","education.num", "capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nAdultTrain["income"]

XTadult = nAdultTest[["age","education.num", "capital.gain", "capital.loss", "hours.per.week"]]
#YTadult = nAdultTest["income"]
XTadult.head()
knn = KNeighborsClassifier(n_neighbors=3)
score = cross_val_score(knn, Xadult, Yadult, cv = 10)
score
knn.fit(Xadult, Yadult)
Ypred = knn.predict(XTadult)
print(Ypred)
from sklearn import preprocessing
numTrain = nAdultTrain.apply(preprocessing.LabelEncoder().fit_transform)
numTest = nAdultTest.apply(preprocessing.LabelEncoder().fit_transform)

Xtrain = numTrain.iloc[:,0:14]
Ytrain = numTrain["income"]

Xtest = numTest.iloc[:,0:14]
numTrain.head()
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
knn.fit(Xtrain, Ytrain)
scores
Ypred = knn.predict(Xtest)
Xtrain = numTrain[["age", "education", "marital.status", "occupation", "race", "sex", "capital.gain", "capital.loss"]]
Ytrain = numTrain["income"]

Xtest = numTest[["age", "education", "marital.status", "occupation", "race", "sex", "capital.gain", "capital.loss"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xtrain, Ytrain)

scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
print (scores)
m = 0
mv = 0
for i in range(1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain, Ytrain)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    k = np.average(scores)
    if k > mv:
        mv = k
        m = i
    print(i,k)

print("melhor:", m)
knn = KNeighborsClassifier(n_neighbors=28)
knn.fit(Xtrain, Ytrain)
Ypred = knn.predict(Xtest)
Ypred
Ypredf = []
j = -1
for x in Ypred:
    j = j + 1
    if Ypred[x] < 1:
        Ypredf.append([j,'<=50K'])
    else:
        Ypredf.append([j,'>50K'])

    
df = pd.DataFrame(Ypredf, columns = ['Id','income'])
df.to_csv("Ypredf.csv", index=False)
df