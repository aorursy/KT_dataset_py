# Este ambiente Python 3 vem com muitas bibliotecas analíticas úteis instaladas
# É definido pela imagem Docker kaggle / python: https://github.com/kaggle/docker-python
# Por exemplo, aqui estão vários pacotes úteis para carregar

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
data=data.drop(columns=["PassengerId"])
data=data.drop(columns=["Category"])
data.columns = ['País', 'Nome', 'Sobrenome', 'Sexo', 'Idade', 'Sobrevivente']
data["Sobrevivente"] = data["Sobrevivente"].replace(to_replace=0, value= "Não")
data["Sobrevivente"] = data["Sobrevivente"].replace(to_replace=1, value= "Sim")
data.head()
print(data.País.describe())
print(data.Idade.describe())
print("Média:", data.Idade.mean())
print("Desvio padrão:", data.Idade.std())
print("Contagem por sexo:\n",data.Sexo.value_counts())
data.groupby(['Sexo']).mean()
data["Sobrevivente"].value_counts().plot.pie(title="Número de Sobreviventes")
data.País.value_counts().plot.bar()
data.Idade.plot.hist()