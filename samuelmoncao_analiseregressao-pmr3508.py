# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import math
import os
print(os.listdir("../input/pmr-data"))
import os
import collections, numpy
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# Any results you write to the current directory are saved as output.
#Aqui, analisaremos a quantidade de linhas e headers dos dados que serão utilizados

totaltrain = pd.read_csv("../input/pmr-data/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?",
        dtype=np.float64)
totaltrain.shape
totaltrain = totaltrain.apply(pd.to_numeric)
#Aqui analisaremos o header dos dados de treino
#Tiraremos as casas com valor acima de 20000 para que os dados fiquem mais consistentes
train = totaltrain.drop(totaltrain[(totaltrain.median_house_value > 200000)].index)
train.head()
#Analisaremos agora a relacao de cada dado com o valor da casa da regiao

#Com isso, identificaremos quais variaveis aparentam uma relacao linear com essa variavel

#Notamos que a varia <longitude> tem uma relacao nao linear com a coluna <median_house_value>
plt.plot(train["longitude"],train["median_house_value"],'ro')
#Notamos que a varia <latitude> tem uma relacao nao linear com a coluna <median_house_value>
plt.plot(train["latitude"],train["median_house_value"],'bo')
#Notamos que a varia <median_age> aparenta nao ter relacao com a coluna <median_house_value>
plt.plot(train["median_age"],train["median_house_value"],'yo')
#Notamos que a varia <total_rooms> tem uma relacao pouco linear com a coluna <median_house_value>
plt.plot(train["total_rooms"],train["median_house_value"],'go')
#Notamos que a varia <total_bedrooms> tem uma relacao pouco linear com a coluna <median_house_value>
plt.plot(train["total_bedrooms"],train["median_house_value"],'r^')
#Notamos que a varia <population> tem uma relacao pouco linear com a coluna <median_house_value>
plt.plot(train["population"],train["median_house_value"],'b^')
#Notamos que a varia <households> tem uma relacao pouco linear com a coluna <median_house_value>
plt.plot(train["households"],train["median_house_value"],'y^')
#Notamos que a varia <median_income> tem uma relacao altamente linear com a coluna <median_house_value>
plt.plot(train["median_income"],train["median_house_value"],'g^')
#Pela analise dos dados, as relacoes que serao usadas inicialmente levarao em conta
#as colunas <latitude>, <longitude> e <median_income> para o regressor KNN

Xtrain = train[["longitude","latitude","median_income"]]
Ytrain = train.median_house_value

#Primeiramente, implementaremos o classificador knn, identificando os melhores parametros
#para um classificador eficaz, que sera usado para analises comaprativas mais a frente
knn = KNeighborsRegressor(n_neighbors=11)
knn.fit(Xtrain, Ytrain)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10,scoring='neg_mean_squared_log_error')

#Com isso, temos:
#1. A qualidade do nosso regressor KNN
math.log10(abs(scores.mean()))
#De acordo com a bibliografia disponibilizada, o ideal seria padronizar os preditores 
#(standardizing the predictors) para que os pesos nao sejam discrepantes. Mas utilizaremos
# a base normal inicialmente
scaler = StandardScaler()
scaler.fit(Xtrain)
X_train_std = scaler.transform(Xtrain)

#Considerando agora as colunas para o regressor Ridge
Xtrain = train[["longitude","latitude","median_age","median_income","households","population","total_rooms","total_bedrooms"]]
reg = linear_model.Ridge(alpha = 0.5)
reg.fit (Xtrain, Ytrain)
scores2 = cross_val_score(reg, Xtrain, Ytrain, cv=10,scoring='neg_mean_squared_log_error')

#2. A qualidade do nosso regressor Ridge
scores2.mean()
#Agora, vamos interar com diversos alphas para que encontremos a influencia dessa variavel
#na determinacao dos coeficientes
n_alphas = 200
alphas = np.logspace(0.01, 20, n_alphas)
coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit (Xtrain, Ytrain)
    coefs.append(ridge.coef_)

#Desenhando o grafico com os pesos de cada variavel
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('pesos')
plt.title('Coeficientes de Ridge em funcao de alpha')
plt.show()
#Considerando agora as colunas para o regressor Lasso
Xtrain = train[["longitude","latitude","median_age","median_income","households","population","total_rooms","total_bedrooms"]]
reg = linear_model.Lasso(alpha = 0.5)
reg.fit (Xtrain, Ytrain)
scores3 = cross_val_score(reg, Xtrain, Ytrain, cv=10,scoring= 'neg_mean_squared_log_error')

#3. A qualidade do nosso regressor Lasso
math.log10(abs(scores3.mean()))

# Notamos que os dois regressores nao possuem uma alta variancia para os dados
#utilizados.
# A variancia é maior no primeiro modelo visto a flexibilidade do modelo KNN. 
# Para os dados utilizados acima, vemos que o menor erro se deu com o regressor Ridge.