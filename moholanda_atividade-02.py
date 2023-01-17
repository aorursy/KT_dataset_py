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
import matplotlib.pyplot as plt                       

casa = pd.read_csv('/kaggle/input/housedata/data.csv') 

casa.head(5)
casa.describe()
casa['price'].max()
casa['price'].min()
casa['bathrooms'].max()
(casa['waterfront'].value_counts())

print('33')
casa['city'].nunique()
cidade = casa['city'].mode()

print(cidade)
import matplotlib.pyplot as plt

casas = casa['city'].value_counts()

plt.hist(casas, bins=10, rwidth = 0.9)
casa.corr()

correlacao=casa['bedrooms'].corr(casa['price'])

print(correlacao)
taxas = []

for n in casa['floors'].value_counts():

    taxas.append(str(round(n/casa['floors'].count()*100,1))+"%")

plt.pie(casa['floors'].value_counts(), labels=taxas)
seattle = casa[(casa['city'] == 'Seattle') & (casa['bedrooms'] == 3.0)]



media = seattle['price'].mean()

print(media)

desvio = seattle['price'].std()

print(desvio)
plt.scatter(casa['bathrooms'],casa['price'])

plt.title("Banehiros x Pço de venda")

plt.xlabel("Banheiros")

plt.ylabel("Pço de venda")

plt.show()

casa.corr()

bp=casa['bathrooms'].corr(casa['price'])

print(bp)
casa['bathrooms'].value_counts().plot.barh()

plt.title("Banheiros x Casas")

plt.xlabel("Casas")

plt.ylabel("Banheiros")