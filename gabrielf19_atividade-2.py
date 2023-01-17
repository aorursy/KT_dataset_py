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
casa.describe()
max = casa.price.max()

min = casa.price.min()



print("valor máximo",max,'e',"valor mínimo",min)
casa.bathrooms.max()

casa.waterfront.sum()
lista = casa['city'].unique()



print(len(lista))
cidade = casa['city'].mode()



print(cidade)

casa['bedrooms'] = casa['bedrooms'].astype(int)

casa['bathrooms'] = casa['bathrooms'].astype(int)

casa['floors'] = casa['floors'].astype(int)

casa['city'] = casa['city'].astype('category')
import matplotlib.pyplot as plt



plt.hist(casa['price'],bins=8, rwidth=0.9)

plt.title("Histograma do número de casas")

plt.xlabel("Cidade")

plt.ylabel("Número de Casa")

plt.grid()



plt.show()
casa.corr()



x = casa ['bedrooms']

y = casa ['price']

df = pd.DataFrame(x, columns = ['bedrooms'])

df['price'] = y

df.head()

df.corr()
import matplotlib.pyplot as plt 



casa['floors'] = casa['floors'].astype(int)



andares = casa["floors"].value_counts()

label_andar=[]



for andar in andares:

    taxa_andar = andar/casa["floors"].count()*100

    label_andar.append(str(round(taxa_andar,2))+"%")

    

plt.figure(figsize=(5,5))

plt.pie(casa["floors"].value_counts(), labels=[label_andar[0],label_andar[1],label_andar[2]])

plt.show()



print(andares)
seattle = casa[(casa['city'] == "Seattle") & (casa['bedrooms'] == 3.0)]



media_preço = seattle['price'].mean()

desvio_padrao = seattle['price'].std() 



print(media_preço,"(Media)")

print(desvio_padrao,"(Desvio)")



from sklearn import datasets as dt



plt.scatter(x = casa['bathrooms'],

            y = casa['price'])

plt.title("Banheiros x Preço")

plt.xlabel("Banheiros")

plt.ylabel("Preço")

plt.show()



casa.corr()

print(casa['bathrooms'].corr(casa['price']))
import matplotlib.pyplot as plt



banheiros = casa['bathrooms'].astype(int).value_counts()



plt.bar(0,banheiros[0], width = 0.8,  color='red')

plt.bar(1,banheiros[1], width = 0.8,  color='blue')

plt.bar(2,banheiros[2], width = 0.8,  color='red')

plt.bar(3,banheiros[3], width = 0.8,  color='blue')

plt.bar(4,banheiros[4], width = 0.8,  color='red')

plt.bar(5,banheiros[5], width = 0.8,  color='blue')

plt.bar(6,banheiros[6], width = 0.8,  color='red')

plt.bar(8,banheiros[8], width = 0.8,  color='red')



plt.title("Distribuição de Banheiro por Casa")

plt.xlabel("Casa")

plt.ylabel("Número de Banheiros")

plt.grid(True)

plt.show()



print(banheiros)