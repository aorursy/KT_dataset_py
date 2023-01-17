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
house = pd.read_csv('/kaggle/input/housedata/data.csv')                                              
house.head()
house.describe()
house['price'].min()

house['price'].max()
house['bathrooms'].max()
house['waterfront'].sum()
house['city'].nunique()
house['city'].max()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

plt.figure (figsize=(20,20))

sns.countplot(data=house, y='city')

plt.show()
corr1 = house[['bedrooms','price']]

corr1.corr()
import matplotlib.pyplot as plt

plt.pie(house['floors'].value_counts(), labels=(1, 2, 3, 4, 5, 6))

plt.show()
housenew = house[(house['bedrooms']==3) & (house['city']=='Seattle')]

housenew.head()

print(housenew.std()) #desvio padrão

print(housenew.mean()) #media
import plotly.offline as py

import plotly.graph_objs as go

trace = go.Scatter(x = house['bathrooms'],

                   y = house['price'],

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
corr2 = house[['bedrooms','price']]

corr2.corr()
nBathroomTypes = house['bathrooms'].nunique()
nBathroomTypes = house['bathrooms'].unique() 

house['bathrooms']=house['bathrooms'].astype(int)

countBathroomTypes = house['bathrooms'].value_counts()

plt.figure(figsize=(15,6))                           



n = 0                                                 

for g in countBathroomTypes:                          

    plt.barh(n, g)                   

    plt.text(g,n, g)                                 

    n += 1

     

plt.title('Distribuição de Banheiros por Casas')

plt.xlabel('Quantidade de Casas')                      

plt.ylabel('Número de Banheiros')                   

plt.legend('cores')                                

plt.show()
house['bathrooms'].max()