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

casas = pd.read_csv('/kaggle/input/housedata/data.csv') 



casas['bedrooms'] = casas['bedrooms'].astype(int)

casas['bathrooms'] = casas['bathrooms'].astype(int)

casas['floors'] = casas['floors'].astype(int)

casas['city']= casas['city'].astype('category')



casas.head() 
casas.describe()
valor_minimo = casas['price'].min()

valor_maximo = casas['price'].max() 

            

str_minimo = "Valor Minimo de Venda = "+str(round(valor_minimo,2)) 

str_maximo = "Valor Maximo de Venda = "+str(round(valor_maximo,2)) 



print(str_minimo)

print(str_maximo)
banheiros = casas['bathrooms'].max() 

print (banheiros)
casas['waterfront'].count()
casas['city'].nunique()
casas.columns
mais_casas = casas['city'].mode()



print('cidade com mais casas : ', mais_casas[0])
plt.figure(figsize=(10,5))

plt.hist(casas['yr_built'], bins=15,rwidth=0.9)



plt.title("histograma do número de casas")



plt.xlabel("ano fabricação")

plt.ylabel("Número construcão")

plt.grid()



plt.show()
casas.shape
from sklearn import datasets as dt

casas = dt.load_iris()

df = pd.DataFrame(casas.data, columns=casas.feature_names)

plt.scatter(df['petal width (cm)'],df['sepal length (cm)'])

plt.title("Petala width x Setala width")

plt.xlabel("petal width")

plt.ylabel("sepal width")

plt.show()

print(casas['floors'].value_counts())

plt.pie((casas['floors'].value_counts()).sort_values(), labels=['3 andares','2 andares','1 andares'])#.sort_values() coloca grafico em ordem crescente

plt.show()
df = pd.DataFrame({'city' : ["Seattle"],

                   'Quartos' : [3],})

df
casas_sea = casas.groupby('city', sort=True)['Seattle'].value_counts()

print(casas_sea)
df = pd.DataFrame({'city' : ["Seattle", "Abbie", "Harry", "Julia", "Carrie"],

                   'Faltas' : [3,4,2,1,4],

                   'Prova' : [2,7,5,10,6],

                   'Seminário': [8.5,7.5,9.0,7.5,8.0]})

df





plt.figure(figsize=(12,10))

banheiros = casas['bathrooms'].value_counts()

print(banheiros)



#plt.bar(x, height, width=0.8, color='red')

plt.bar(0,banheiros[0], width = 0.9,  color='blue')

plt.bar(1,banheiros[1], width = 0.9,  color='yellow')

plt.bar(2,banheiros[2], width = 0.9,  color='green')

plt.bar(3,banheiros[3], width = 0.9,  color='blue')

plt.bar(4,banheiros[4], width = 0.9,  color='yellow')

plt.bar(5,banheiros[5], width = 0.9,  color='green')

plt.bar(6,banheiros[6], width = 0.9,  color='blue')

#plt.bar(0,banheiros[0], width = 0.9,  color='yellow')

plt.bar(8,banheiros[8], width = 0.9,  color='green')



plt.title('Gráfico : Casas X Banheiros')



plt.xticks([0,1,2,3,4,5,6,7,8],['0 banheiros','1 banheiros','2 banheiros','3 banheiros','4 banheiros','5 banheiros','6 banheiros ','7 banheiros','8 banheiros'])

#plt.legend(['0','1','2','3'])

plt.show()