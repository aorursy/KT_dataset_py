# Nome: Douglas de Sousa dos Santos  317103457



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt                        #importando a biblioteca gráfica

dados = pd.read_csv('/kaggle/input/housedata/data.csv')               
dados.info()

print(dados)
print(dados['price'].max())

print(dados['price'].min())
dados['bathrooms'].max()
dados['waterfront'].count()
(dados['city'].value_counts()).count()
dados['city'].mode()
contagem = dados['city'].value_counts()

cidades = contagem.index

#print(contagem , cidades)



plt.figure(figsize=(35,20))

for n, i in enumerate(contagem):

    plt.barh(n,i)

    plt.text(i+20,n,str(cidades[n]) + ' ' + str(i))
dados.corr()
taxa_andar = dados['floors'].value_counts()

print(taxa_andar)

total = dados['floors'].count()

grafico = [taxa_andar[1]/total*100,taxa_andar[2]/total*100,taxa_andar[3]/total*100]

plt.pie(grafico, labels=[round(taxa_andar[1]/total*100,2),round(taxa_andar[2]/total*100,2),round(taxa_andar[3]/total*100,2)])

print(grafico)

dados1 = dados[(dados['city']=='Seattle') & (dados['bedrooms']==3)]



print(dados1.std())



media = dados1.median()



print(media)
dados.corr()
dados['bathrooms'] =  dados['bathrooms'].astype(int)

contagem = dados['bathrooms'].value_counts().sort_index()

banheiros = contagem.index

print(contagem , banheiros)



plt.figure(figsize=(8,8))

for n, i in enumerate(contagem):

    plt.barh(n,i)

    plt.text(i+20,n,str(i))

plt.yticks([0,1,2,3,4,5,6,8])