import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import seaborn as sns
df = pd.read_csv("../input/kitnets_cuiaba.csv", encoding = "ISO-8859-1")

df.head()
preco = df[['Valor','Tamanho']]

preco.head()
sns.pairplot(data=preco, kind="reg")
from sklearn import linear_model

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

regressao = linear_model.LinearRegression()
X = np.array(preco['Tamanho']).reshape(-1, 1)

y = le.fit_transform(preco['Valor'])

regressao.fit(X, y)
tamanho = 40

print('Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))
tamanhos = [20,25,30,35,40,45]

for i in tamanhos:

    j = regressao.predict(np.array(i).reshape(-1, 1))

    print('Tamanho: ',i,' Valor: ',j,'\n')
precoQuartos = df[['Descricao','Tamanho','Valor']]

precoQuartos.head()
X = np.array(precoQuartos['Tamanho']).reshape(-1, 1)

y = le.fit_transform(precoQuartos['Valor'])

regressao.fit(X, y)
tamanho = 55

print('Tamanho: ',tamanho, 'Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))