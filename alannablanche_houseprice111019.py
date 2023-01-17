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

casa = pd.read_csv("../input/housedata/data.csv")

casa.head()                                                
casa.describe()
import matplotlib.pyplot as plt

casa = pd.read_csv("/kaggle/input/housedata/output.csv")
print(casa['price'].max())

print(casa['price'].min())
casa['bedrooms'] = casa['bedrooms'].astype(int)

casa['bathrooms'] = casa['bathrooms'].astype(int)

casa['floors'] = casa['floors'].astype(int)

#casa['city'] = casa['city'].astype('category')

print(casa['bathrooms'].max())
print(casa['waterfront'].nunique())
print(casa['city'].nunique())
casa['city'].mode()
casa.shape
corr1 = casa[['bedrooms','price']]

corr1.corr()
import pandas as pd 

from sklearn import datasets as dt

casa1 = dt.load_data()

df = pd.DataFrame(casa.data, columns=casa.feature_names)

plt.scatter(df['bedrooms'],df['price'])

plt.title("quartos x preço.")

plt.xlabel("bedrooms")

plt.ylabel("price")

plt.show()
print(casa['floors'].value_counts())

plt.pie(casa["floors"].value_counts(), colors=['purple', 'grey','red'], labels=[1,2,3])

plt.show()

conta = casa["floors"].value_counts()

and1 = conta[0]/casa["floors"].count()*100

and2 = conta[1]/casa["floors"].count()*100

and3 = conta[2]/casa["floors"].count()*100



andar1 = "1 Andar" + str(round(and1,2))+"%"

andar2 = "2 Andares" + str(round(and2,2))+"%"

andar3 = "3 Andares" + str(round(and3,2))+"%"

plt.figure(figsize=(5,5))

plt.pie(casa["floors"].value_counts(), labels=[andar1,andar2,andar3])

plt.show()
casanew = casa[(casa['bedrooms']==3) & (casa['city']=='Seattle')]

casanew.head()

print(casanew.std()) #desvio padrão

print(casanew.mean()) #media
import plotly.offline as py

import plotly.graph_objs as go

trace = go.Scatter(x = casa['bathrooms'],

                   y = casa['price'],

                   mode = 'markers')

data = [trace]

py.iplot(data)

layout = go.Layout(title='Numero de banheiros x Preço de venda',

                   yaxis={'title':'Preço da casa'},

                   xaxis={'title': 'Banheiros'})

# Criando figura que será exibida

fig = go.Figure(data=data, layout=layout)

# Exibindo figura/gráfico

py.iplot(fig)
nBathroomTypes = casa['bathrooms'].unique() 

casa['bathrooms']=casa['bathrooms'].astype(int)

countBathroomTypes = casa['bathrooms'].value_counts()

plt.figure(figsize=(15,6))                           



n = 0                                                 

for g in countBathroomTypes:                          

    plt.barh(n, g)                   

    plt.text(g,n, g)                                 

    n += 1

     

plt.title('Distribuição de Banheiros por Casas')

plt.xlabel('Quantidade de Casas')                      

plt.ylabel('Número de Banheiros')                   

                           

plt.show()
total = casa.groupby ['bathrooms'].value_counts()

print(total)

plt.bar(0,total[1], width=0.5, color='red')

plt.bar(1,total[0], width=0.5, color='purple')

plt.bar(2,total[2], width=0.5, color='blue')

plt.show()