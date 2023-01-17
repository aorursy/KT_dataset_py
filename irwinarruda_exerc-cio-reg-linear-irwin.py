#importar as bibliotecas de matrizes e manipulação de dados

import numpy as np

import pandas as pd
#importar bibliotecas de plotar

import matplotlib.pyplot as plt
#importar bibliotecas machineLearning

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDRegressor #Stochastic Gradient Descedent
#deletar arquivos

import os

#os.remove("/kaggle/working/")
#carregar o dataset

dataset = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")
dataset.head(5)
dataset.describe()
#plotar gráfico total

plt.scatter(dataset[["weight"]], dataset[["mpg"]])

plt.xlabel("peso(libras)")

plt.ylabel("Autonomia (mpg)")

plt.title("Relação entre peso e Autonomia")

#plt.savefig("./fig123123.png")

#dataset.loc([1][2])
X = dataset[["weight"]]

y = dataset[["mpg"]]

#lbs para kg

X *= 0.453592

#m/g para km/l

y *= 0.425144
#sormalizar os dados de X

escala = StandardScaler()

X_norm = escala.fit_transform(X)
#separar dados para treino e teste

X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)
#indicar os parâmetro da regressão

reglinear = SGDRegressor(max_iter=500, eta0=0.01, tol=0.0001, learning_rate="constant", verbose=1)

reglinear.fit(X_norm_train, y_train)
#prever a autonomia nos casos teste separados

y_pred_test = reglinear.predict(X_norm_test)

X_test = escala.inverse_transform(X_norm_test)
#plotar o gráfico dos casos teste

plt.scatter(X_test, y_test, label="Real")

plt.scatter(X_test, y_pred_test, label="Previsto")

plt.xlabel("peso(kg)")

plt.ylabel("autonomia(km/l)")

plt.title("Relação entre autonomia e o peso")

plt.legend(loc=1)
#já conferido, prever os dados de forma geral

y_pred = reglinear.predict(X_norm)
#plotar o gráfico real, com a previsão feita pela regressão linear

plt.scatter(X, y, label="Real")

plt.scatter(X, y_pred, label="Previsto")

plt.xlabel("peso(kg)")

plt.ylabel("autonomia(km/l)")

plt.title("Relação entre autonomia e peso")

plt.legend(loc=1)