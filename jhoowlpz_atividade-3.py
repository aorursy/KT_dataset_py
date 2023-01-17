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

casa.head()



casa['bedrooms'] = casa['bedrooms'].astype(int)

casa['bathrooms'] = casa['bathrooms'].astype(int)

casa['floors'] = casa['floors'].astype(int)

casa['city'] = casa['city'].astype('category')
casa.describe()
min = casa['price'].min()

max = casa['price'].max()



print("Valor Máximo:", max)

print("Valor Mínimo:", min)
max = casa['bathrooms'].max()



print("O Maior Número de Banheiros é:", max)

resultado = casa['waterfront'].value_counts()



print(resultado[1],"casas!")
resultado = casa['city'].nunique()



print(resultado,"casas!")
resultado = casa['city'].mode()



print(resultado[0],"é a cidade com mais casas!")


num_casas = casa['price'].count()   

str_num = "Numero de Casas ="+str(round(num_casas,2)) 



plt.hist(casa['price'],bins=4, rwidth=0.8)           

plt.title('Histograma do Número de Casas')

plt.xlabel('Casa')

plt.ylabel('Quantidade')    

plt.show()
casa.corr()


x = casa['bedrooms']

y = casa['price']

df = pd.DataFrame(x, columns=['bedrooms'])

df['price']=y

df.head()

df.corr()





contagem = casa["floors"].value_counts()

label_name=[]



for cont in contagem:



    taxa_andar = cont/casa["floors"].count()*100

    label_name.append(str(round(taxa_andar,2))+"%") 

    

plt.figure(figsize=(6,6))

plt.pie(casa["floors"].value_counts(), labels=[label_name[0],label_name[1],label_name[2]])

plt.show()
seatle = casa[(casa['city']== 'Seattle') & (casa['bedrooms'] == 3.0)]



media_preco = seatle['price'].mean()

desvio_padrao = seatle['price'].std()



print(media_preco)



print(desvio_padrao)



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



banheiros = casa['bathrooms'].value_counts()



plt.bar(0,banheiros[0], width = 0.8, color='red')

plt.bar(1,banheiros[1], width = 0.8, color='blue')

plt.bar(2,banheiros[2], width = 0.8, color='red')

plt.bar(3,banheiros[3], width = 0.8, color='blue')

plt.bar(4,banheiros[4], width = 0.8, color='red')

plt.bar(5,banheiros[5], width = 0.8, color='blue')

plt.bar(6,banheiros[6], width = 0.8, color='red')

plt.bar(8,banheiros[8], width = 0.8, color='blue')



plt.title("Distribuição de Banheiros por Casa")

plt.xlabel("Casa")

plt.ylabel("Número de Banheiros")

plt.grid(True)

plt.show()