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
casa.describe()                                   
casa.head()
casa.shape
min = casa['price'].min()

max = casa['price'].max()

print("O Valor mínimo de venda é",min)

print("O valor máximo de venda é",max)
bathroom = casa['bathrooms'].max()

print("O maior número de banheiros é",bathroom)
waterf = casa['waterfront'].sum()

print(waterf,"casas.")
dif = casa['city'].nunique()

print("O dataset possui casas em",dif,"cidades diferentes.")
maisc = casa['city'].mode()

valor = casa['city'].value_counts()

maior = valor.max()

print(maisc,"com",maior,"casas.")
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

plt.hist(casa['yr_built'])

plt.title("Histograma do Número de Casas")

plt.xlabel("Contagem")

plt.ylabel("Número de Casas")

plt.grid()

plt.show()
plt.scatter(casa['bedrooms'],casa['price'])

plt.title("Quartos x Preço")

plt.xlabel("quartos")

plt.ylabel("preço (m)")

plt.show()
contagem = casa["floors"].value_counts()



andar1 = contagem[1]/casa["floors"].count()*100

andar2 = contagem[2]/casa["floors"].count()*100

andar3 = contagem[3]/casa["floors"].count()*100



um_and = "1 andar" + str(round(andar1,2))+"%"

dois_and = "2 andares" + str(round(andar2,2))+"%"

tres_and = "3 andares" + str(round(andar3,2))+"%"



plt.figure(figsize=(5,5))

plt.pie(casa["floors"].value_counts(),labels=[1,2,3,4,5,6])

plt.show()

plt.scatter(casa['bathrooms'],casa['price'])

plt.title("banheiros x Preço")

plt.xlabel("banheiros")

plt.ylabel("preço (m)")

plt.show()
plt.figure(figsize=(10,8))

plt.hist(casa['bathrooms'], bins=10, width=0.7)

plt.title("Histograma do Número de banheiros")

plt.xlabel("Contagem")

plt.ylabel("Número de banheiros")

plt.grid()

plt.show()
casa['bedrooms'] = casa['bedrooms'].astype(int)

casa['bathrooms'] = casa['bathrooms'].astype(int)

casa['floors'] = casa['floors'].astype(int)

casa['city'] = casa['city'].astype('category')