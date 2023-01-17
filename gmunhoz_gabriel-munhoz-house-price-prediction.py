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

dados = pd.read_csv('/kaggle/input/housedata/data.csv') 

dados.head(100)                                  
dados['bedrooms'] = dados['bedrooms'].astype(int)

dados['floors'] = dados['floors'].astype(int)

dados['bathrooms'] = dados ['bathrooms'].astype(int)

dados['city'] = dados ['city'].astype('category')
dados.describe()
print(dados['price'].min())

print(dados['price'].max())
print(dados['bathrooms'].max())
dados['waterfront'].count()
dados['city'].nunique()
cidade = dados['city'].mode()

print('A cidade com mais casas é: ', cidade)
dados['city'].describe()
contagem = dados['city'].value_counts()

cidades = contagem.index



print(contagem)

print(cidades)



plt.figure(figsize=(20,10))



for n, i in enumerate(contagem):

    plt.barh(n,i)

    plt.text(i+20,n, cidades[n])



plt.ylabel("Número de casas")

plt.xlabel('Cidades')

   

plt.grid()

plt.show()
x = dados['price'].corr(dados['bedrooms'])

print(round(x,2))
contagem = dados['floors'].value_counts()



print(contagem)



um_andar = contagem[1]/dados["floors"].count()*100

dois_andares = contagem[2]/dados["floors"].count()*100

tres_andares = contagem[3]/dados["floors"].count()*100



str_um_andar = "1 Andar " + str(round(um_andar,1))+"%"

str_dois_andares = "2 Andares " + str(round(dois_andares,2))+"%"

str_tres_andares = "3 Andares " + str(round(tres_andares,3))+"%"



plt.figure(figsize=(5,5))

plt.pie(dados["floors"].value_counts(), labels=[str_um_andar,str_dois_andares,str_tres_andares])

plt.pie(dados["floors"].value_counts())



plt.show()
seatle_houses = dados[(dados['bedrooms']==3) & (dados['city']=='Seattle')]

seatle_houses.head()



print(seatle_houses.std()) 

print(seatle_houses.mean())
import plotly.offline as py

import plotly.graph_objs as go



house = go.Scatter(x = dados['bathrooms'],

                   y = dados['price'],

                   mode = 'markers')

data = [house]

py.iplot(data)

layout = go.Layout(title='Numero de banheiros x Preço de venda',

                   yaxis={'title':'Preço da casa'},

                   xaxis={'title': 'Banheiros'})



# Criando figura que será exibida

fig = go.Figure(data=data, layout=layout)

# Exibindo figura/gráfico

py.iplot(fig)
contagem = dados['bathrooms'].value_counts()

cidades = contagem.index



print(contagem)

print(cidades)



plt.figure(figsize=(15,10))



for n, i in enumerate(contagem):

    plt.barh(n,i)

    plt.text(i+20,n, cidades[n])



plt.ylabel("Banheiros")

plt.xlabel('Casas')

   

plt.grid()

plt.show()