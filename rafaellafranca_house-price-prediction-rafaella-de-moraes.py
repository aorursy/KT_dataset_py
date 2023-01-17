# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
house=pd.read_csv ('/kaggle/input/housedata/data.csv')               
house.head()
min=house['price'].min()

max=house['price'].max()

print(min,max)
house['bathrooms'].max()
house['waterfront'].sum()
house['city'].nunique()
house['city'].mode()
import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))

plt.hist(house.count(),bins=10)

plt.title("Histograma número de casas")

plt.xlabel("Date")

plt.ylabel("Número de casas")

plt.grid()



plt.show()
house.plot(x='bedrooms',y='price',kind='scatter', title='Bedrooms x Price',color='r')
contagem=house['floors'].value_counts()



andar1=contagem[1]/house['floors'].count()*100

andar2=contagem[2]/house['floors'].count()*100

andar3=contagem[3]/house['floors'].count()*100



um_and='1 andar' + str(round(andar1,2))+'%'

dois_and='2 andares' + str(round(andar2,2))+'%'

tres_and='3 andares' + str(round(andar3,2))+'%'



plt.figure(figsize=(5,5))

plt.pie(house['floors'].value_counts(),labels=[1,2,3,4,5,6])

plt.show()
import pandas as pd

df = pd.DataFrame()

df

Coluna = [

    'Seatle']



df = pd.DataFrame(columns=Coluna)

df

Columns: []

Index: []

df

plt.scatter(house['bedrooms'],house['price'])

plt.title("Quartos x Preço")

plt.xlabel("bedrooms")

plt.ylabel("price")

plt.show()
nanos=house['bathrooms'].nunique()

banheiro=house['bathrooms'].unique()

num_min=house['bathrooms'].min()

num_max=house['bathrooms'].max()

count_banheiro=house['bathrooms'].value_counts().sort_index()



print(nanos)

print(banheiro)

print(num_min)

print(num_max)

print(count_banheiro)
plt.figure(figsize=(15,10))

plt.hist(house['bathrooms'],bins=nanos, rwidth=0.8)

plt.title("Distribuição dos Banheiros por Casa")

plt.xlabel("bathrooms")

plt.ylabel("House")

plt.grid()

plt.show()