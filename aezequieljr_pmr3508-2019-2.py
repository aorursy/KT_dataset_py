# Importando o que utilizaremos no programa

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
# Realizando leitura da base de treinamento

trainAdult = pd.read_csv("../input/adultdb/train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")
# Formato da base

trainAdult.shape
# Estrutura da base

trainAdult.head()
# Retirando linhas com dados faltantes

trainAdult = trainAdult.dropna()
trainAdult.shape
# Idade

trainAdult["age"].plot(kind='hist',bins=10);
# Etnia

trainAdult['race'].value_counts().plot(kind="pie")
# País

trainAdult["native.country"].value_counts()
# Gênero

trainAdult["sex"].value_counts().plot(kind="bar")
# Ocupação

trainAdult["occupation"].value_counts().plot(kind="bar")
# Renda

trainAdult["income"].value_counts().plot(kind="bar")
# Setando os inputs e outputs da base de treino

XtrainAdult = trainAdult[["age","education.num", "capital.gain", "capital.loss", "hours.per.week"]]

YtrainAdult = trainAdult.income
# Realizando leitura da base de teste

testAdult = pd.read_csv("../input/adultdb/test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")

#testAdult = testAdult.dropna()

XtestAdult = testAdult[["age","education.num", "capital.gain", "capital.loss", "hours.per.week"]]
# kNN - Treinando o modelo

kNN = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(kNN, XtrainAdult, YtrainAdult, cv=10)

kNN.fit(XtrainAdult, YtrainAdult)

scores
# kNN - Aplicando o modelo

YtestAdult = kNN.predict(XtestAdult)

print(len(testAdult.index.values), len(YtestAdult))
# Enviando resultados

result = np.vstack((testAdult["Id"], YtestAdult)).T

x = ["Id","income"]

Resultado = pd.DataFrame(columns = x, data = result)

Resultado.to_csv("Resultado.csv", index = False)