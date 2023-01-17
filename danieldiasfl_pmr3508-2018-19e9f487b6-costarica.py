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
AdultTrain = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

# DADOS DE TESTE
AdultTest = pd.read_csv('../input/test.csv',
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

AdultTrain.shape
# nadult - tira linhas com dados faltantes
nAdultTrain = AdultTrain.dropna()
nAdultTest = AdultTest.dropna()

nAdultTrain.shape
AdultTrain.isnull()
AdultTrain2 = AdultTrain.drop(["Id","v2a1", "v18q1", "rez_esc"], axis = 1)
AdultTest2 = AdultTest.drop(["Id","v2a1", "v18q1", "rez_esc"], axis = 1)

AdultTrain2.head()
# nadult - tira linhas com dados faltantes
nAdultTrain = AdultTrain2.dropna()
nAdultTest = AdultTest2.dropna()

nAdultTrain.shape
#transforma tudo em numero
from sklearn import preprocessing
numTrain = nAdultTrain.apply(preprocessing.LabelEncoder().fit_transform)
numTest = nAdultTest.apply(preprocessing.LabelEncoder().fit_transform)
#para x usar todas as colunas menos o target:
Xtrain = numTrain.drop("Target", axis=1)
#seleciona o target
Ytrain = numTrain["Target"]

Xtest = numTest
Xtrain.shape, AdultTrain.shape
m = 0
mv = 0
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain, Ytrain)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv=5)
    k = np.average(scores)
    if k > mv:
        mv = k
        m = i
    print(i,k)

print("melhor:", m)
knn = KNeighborsClassifier(n_neighbors=33)
knn.fit(Xtrain, Ytrain)
score = cross_val_score(knn, Xtrain, Ytrain, cv = 5)
score.mean()
Ypred = knn.predict(Xtest)