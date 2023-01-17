# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
casa = pd.read_csv('/kaggle/input/housedata/data.csv')





casa.head()
min = casa['price'].min()

max = casa['price'].max()

print("O valor minimo da casa é $",min)

print("O valor máxmo da casa é $",max)
max = casa['bathrooms'].max()

print("O maior número de banheiros é",max)
casa['waterfront'].sum()
casa['city'].nunique()
mais = casa['city'].mode()

#M = casa['city'].mode()



print(mais)



#print ('A cidade com mais casas é ',casa['city'].value_counts())



import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

plt.hist(casa['yr_built'], bins=10)

plt.title("Distribuição de Preços dos Veículos")

plt.xlabel("Ano")

plt.ylabel("Número de Casas")

plt.grid()



plt.show()
casa.plot(x='bedrooms',y='price',kind='scatter', title='Bedrooms x Price',color='r')
contagem=casa['floors'].value_counts()



andar1=contagem[1]/casa['floors'].count()*100

andar2=contagem[2]/casa['floors'].count()*100

andar3=contagem[3]/casa['floors'].count()*100



um_and='1 andar' + str(round(andar1,2))+'%'

dois_and='2 andares' + str(round(andar2,2))+'%'

tres_and='3 andares' + str(round(andar3,2))+'%'



plt.figure(figsize=(5,5))

plt.pie(casa['floors'].value_counts(),labels=[1,2,3,4,5,6])

plt.show()

import pandas as pd

df = pd.DataFrame()

df

Coluna = [

    '']



df = pd.DataFrame(columns=Coluna)

df

Columns: []

Index: []

df

plt.scatter(casa['bedrooms'],casa['price'])

plt.title("Quartos x Preço")

plt.xlabel("bedrooms")

plt.ylabel("price")

plt.show()
nanos=casa['bathrooms'].nunique()

banheiro=casa['bathrooms'].unique()

num_min=casa['bathrooms'].min()

num_max=casa['bathrooms'].max()

count_banheiro=casa['bathrooms'].value_counts().sort_index()



print(nanos)

print(banheiro)

print(num_min)

print(num_max)

print(count_banheiro)