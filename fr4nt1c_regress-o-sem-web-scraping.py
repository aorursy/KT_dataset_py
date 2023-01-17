import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

import seaborn as sns
df = pd.read_csv("../input/sao-paulo-properties-april-2019.csv")

df.head()
preco = df[['Price','Size']]

preco.head()
sns.pairplot(data=preco, kind="reg")
from sklearn import linear_model

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

regressao = linear_model.LinearRegression()
x = np.array(preco['Size']).reshape(-1, 1)

y = le.fit_transform(preco['Price'])

regressao.fit(x, y)
tamanho = 70

print('Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))
tamanhos = [45,50,60,90,120,170,240]

for i in tamanhos:

    j = regressao.predict(np.array(i).reshape(-1, 1))

    print('Tamanho: ',i,' Valor: ',j,'\n')
precoQuartos = df[['Price','Rooms','Size']]

precoQuartos.head()
x = np.array(precoQuartos[['Size','Rooms']])

y = le.fit_transform(preco['Price'])

regressao.fit(x, y)
quartos = 1

tamanho = 70

print('Tamanho: ',tamanho,'Quartos: ',quartos,' Valor: ',regressao.predict(np.array([[tamanho,quartos]])))