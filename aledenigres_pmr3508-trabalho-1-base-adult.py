# Importação de bibliotecas
import pandas as pd
import sklearn
import numpy as np
import os
os.listdir('../input/')
# Importação da base Adult (de treino)
adult = pd.read_csv("../input/adultdataset/train_data.csv",
        engine='python',
        na_values="?")
adult.shape # Dimensões da tabela da base de dados (linhas, colunas)
adult.head() # Início da tabela
adult.describe()
adult["native.country"].value_counts() 
# Importação da biblioteca de plotagem de gráficos
import matplotlib.pyplot as plt
adult["age"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
adult["relationship"].value_counts().plot(kind="pie")
# Retirando linhas com dados faltantes:
nadult = adult.dropna()
nadult.shape # Dimensões da tabela sem as linhas com dados faltantes
TestAdult = pd.read_csv("../input/adultdataset/test_data.csv",
            engine='python',
            na_values="?")
nTestAdult = TestAdult.dropna()
X_adult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Y_adult = nadult.income
X_TestAdult = nTestAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=30)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, X_adult, Y_adult, cv=10)

scores
knn.fit(X_adult,Y_adult)
Y_TestPred = knn.predict(X_TestAdult)

Y_TestPred
result = np.vstack((nTestAdult["Id"], Y_TestPred)).T
x = ["id","income"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("results.csv", index = False)
Resultado