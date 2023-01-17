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
casa = pd.read_csv('/kaggle/input/housedata/data.csv')

casa.head()
casa.shape
casa.describe()
casa.nunique()
casa.describe()

casa.columns
print('minimo : ', casa['price'].min())

print('máximo : ', casa['price'].max())
print('maior qtde banheiros : ', casa['bathrooms'].max())
print('vista para o mar : ', casa['waterfront'].sum(), ', casas')
print('numeros de casa : ', casa['city'].nunique())

  
mais = casa['city'].mode()



print('mais casas : ', mais[0])
plt.figure(figsize=(15,8))

plt.hist(casa['yr_built'], bins=15)

plt.title("Distribuição por data")

plt.xlabel("ano dde fabricação")

plt.ylabel("Número de construcão")

plt.grid()



plt.show()
casa.columns
andares = casa["floors"].value_counts()

#print(andares)



um = andares[1]/casa["floors"].count()*100

dois = andares[2]/casa["floors"].count()*100 

tres = andares[3]/casa["floors"].count()*100

#print(um,dois,tres)



labels = 'um', 'dois', 'tres'

sizes = [um,dois, tres]

explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
pd.value_counts(casa['floors'])
 

quartos = pd.value_counts(casa['bedrooms']==3 & casa['city'] =='seattle')

quartos_3 = quartos[1]

media = round(quartos_3.sum()/casa['bedrooms'].count()*100,2)



print(quartos_3.std())

print('media tres quartos : ',media, '%')

 
tres_quartos = pd.value_counts(casa['bedrooms']==3)

print('casas com 3 quartos : ', tres_quartos[1])
plt.figure(figsize=(5,5))

plt.hist(casa['bathrooms'], bins=8)

plt.title("Distribuição de banheiros por data")

plt.xlabel("quantidade")

plt.ylabel("números de casa")

plt.grid()



plt.show()