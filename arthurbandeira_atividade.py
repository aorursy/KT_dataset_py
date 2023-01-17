# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dados = pd.read_csv('/kaggle/input/housedata/data.csv')

dados.head()
dados.describe()
print(dados['price'].max())

print(dados['price'].min())
print(dados['bathrooms'].max())
dados['waterfront'].value_counts()
dados['city'].nunique() #retorna o número de combustíveis diferentes na coluna
dados['city'].value_counts().max()

import matplotlib.pyplot as plt



plt.hist(dados['city'])

plt.xlabel('Casas')

plt.ylabel('Numero de casas')



plt.show()
from sklearn import datasets as dt

dados = dt.load_iris()

df = pd.DataFrame(dados.data, columns=dados.feature_names)

plt.scatter(df['bedrooms'],df['price'])

plt.title("Quartos x Preço")

plt.show()
contagem =  dados['condition'].value.counts()



primeiro = contagem[0]/dados['condition'].count()*100

segundo = contagem[1]/dados['condition'].count()*100

terceiro = contagem[1]/dados['condition'].count()*100



str_primeiro = "Primeiro andar " + str(round(primeiro,2))+"%"

str_segundo = "Segundo andar " + str(round(segundo,2))+"%"

str_terceiro = "Terceiro andar " + str(round(terceiro,2))+"%"



plt.figure(figsize=(5,5))

plt.pie(carros["condition"].value_counts(), labels=[str_primeiro,str_segundo, str_terceiro])

plt.show()