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
casas = pd.read_csv('/kaggle/input/housedata/data.csv')

casas.head()   
casas['bathrooms']=casas['bathrooms'].astype(int)
casas.describe()
print(casas['price'].min())

print(casas['price'].max())
print(casas['bathrooms'].max())
count = 0

for i in range(casas['waterfront'].count()):

    if casas['waterfront'][i] == 1:

        count = count+1

print(count)

cidades = casas["city"].nunique()

print(cidades)
dados = casas['city'].mode()

print(dados)
contagem = casas['city'].value_counts()

cidades = contagem.index

plt.figure(figsize=(20,20))



for n, i in enumerate(contagem):

    plt.barh(n,i)

    plt.text(i+20,n,str(cidades[n]+' '+ str(i) ))



plt.title("Histograma Número Casas")

plt.xlabel("cidade")

plt.ylabel("contagem")

plt.show()
casas.corr()

correlacao=casas['bedrooms'].corr(casas['price'])

print(correlacao)

#print(casas['price'].corr(casas['bedrooms']))
taxas = []

for n in casas['floors'].value_counts():

    taxas.append(str(round(n/casas['floors'].count()*100,1))+"%")



plt.pie(casas['floors'].value_counts(), labels=taxas)
seattle = casas[(casas['city'] == 'Seattle') & (casas['bedrooms'] == 3.0)]

#print(seattle.head())



media = seattle['price'].mean()

print(media)



desvio_padrao = seattle['price'].std()

print(desvio_padrao)

import matplotlib.pyplot as plt

 

plt.scatter(x = casas['bathrooms'],

            y = casas['price'])

plt.title("Banheiros x Preço")

plt.xlabel("banheiros")

plt.ylabel("preço")

plt.show()



casas.corr()

print(casas['bathrooms'].corr(casas['price']))

#contagem = casas['bathrooms'].value_counts()

#banheiros = contagem.index

#plt.xlim(0, 2500)

#plt.ylim(0, 10)

plt.figure(figsize=(5,5))

plt.barh(casas['bathrooms'], casas['price'])

plt.title("Banheiros x Casas")

plt.xlabel("casas")

plt.ylabel("banheiros")

plt.show()