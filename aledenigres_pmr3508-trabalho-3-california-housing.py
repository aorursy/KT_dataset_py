# BIBLIOTECAS IMPORTADAS

import numpy as np # linear algebra
import pandas as pd # data processing

import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Leitura da base no formato csv
train = pd.read_csv('../input/train.csv',engine='python')
# Base de dados
train
# Tamanho da base
train.shape
# Estatísticas descritivas da base
train.describe()
# Distribuição do valor médio das casas por região (longitude x latitude)
plt.scatter(train["longitude"],train["latitude"],c=train["median_house_value"],s=7)
# Quantidade de regiões por faixa de população
train["population"].value_counts().plot(kind="hist")

# Quantidade de regiões por faixa de casas
train["households"].value_counts().plot(kind="hist")
# Quantidade de regiões por faixa de quartos
train["total_bedrooms"].value_counts().plot(kind="hist")
# Atributos
Xtrain = train.drop(columns=["median_house_value"])
Xtrain.head()
# Variável-alvo (de interesse)
Ytrain = train["median_house_value"]
Ytrain.head()
ridge = linear_model.Ridge(alpha = 0.5)
ridge.fit(Xtrain,Ytrain)
# Coeficientes para cada atributo (na ordem das colunas da base)
ridge.coef_
# Termo independente (intersecção no eixo y)
ridge.intercept_ 
# Pontuação
cross_val_score(ridge,Xtrain,Ytrain,cv=10).mean()
lasso = linear_model.Lasso(alpha = 0.1)
lasso.fit(Xtrain,Ytrain)
# Coeficientes para cada atributo (na ordem das colunas da base)
lasso.coef_
# Termo independente (intersecção no eixo y)
lasso.intercept_ 
# Pontuação
cross_val_score(lasso,Xtrain,Ytrain,cv=10).mean()
kNN = KNeighborsRegressor(n_neighbors=52)
kNN.fit(Xtrain, Ytrain) 
# Pontuação
cross_val_score(kNN, Xtrain, Ytrain, cv=10).mean()
# Leitura da base de teste
test = pd.read_csv("../input/test.csv")
test.head()
test_pred = ridge.predict(test)

for i in range(0,len(test_pred)):
    if test_pred[i] < 0:
        test_pred[i] = np.mean(test_pred)
        
pred = pd.DataFrame(test.Id)
pred["median_house_value"] = test_pred
pred.head()
# Exportação para CSV
pred.to_csv("prediction.csv", index=False)