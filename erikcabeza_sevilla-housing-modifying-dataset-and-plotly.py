import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

 
import numpy as np

import pandas as pd

import seaborn as sns

import plotly.express as px


dataset=pd.read_excel('/kaggle/input/sevilla-housing/Sevilla_housing.xlsx', index_col=0)  
dataset.info()
dataset.reset_index(level=0, inplace=True)
dataset.head()
dataset.tittle.unique()
dataset.drop_duplicates( keep="last") 
#price has to be multiplied by 1000, as it was indicated by the description of the dataset

dataset['price'] = dataset['price'].apply(lambda x: x*1000)

#year of construction was 0 if there wasnt any information-gonna put instead the median 

dataset['year']=dataset['year'].replace(0,dataset['year'].median())

dataset.tittle.unique()
Piso  = ['piso']

Chalet_adosado  = ['chalet adosado']

Chalet_pareado = ['chalet pareado']

Casa = ['casa o chalet independiente']# casa o chalet independiente are the same so that's why I'm calling both casa

Dúplex = ['dúplex']

Cortijo = ['cortijo']

Estudio = ['estudio']

Palacio = ['palacio']

Ático = ['ático']
d = {'Piso' : Piso,

     'Ático' : Ático,

     'Chalet_adosado' : Chalet_adosado,

     'Chalet_pareado': Chalet_pareado,

     'Casa': Casa,

      'Dúplex': Dúplex,

    'Cortijo': Cortijo,

     'Estudio':Estudio,

      'Palacio': Palacio}



d1 = {k: oldk for oldk, oldv in d.items() for k in oldv}



for k, v in d1.items():

    dataset.loc[dataset['tittle'].str.contains(k, case=False), 'type'] = v
dataset.type.unique()
dataset['neighbourhood']=dataset.tittle.str.extract('-(.*)')   
dataset.head()
dataset['neighbourhood'] = dataset['neighbourhood'].str.replace('Sevilla', '')

dataset['neighbourhood'] = dataset['neighbourhood'].str.replace(',', '')
dataset.head()
#With this line of code I will rename the columns tittle, meters and hotels for making the dataset more understandable

dataset.columns = ['description', 'rooms' , 'price', 'size', 'bathrooms', 'garage', 'terrace', 'zipcode', 'year', 'hotelsNear', 'type', 'neighbourhood']
dataset.info()
#because I have NaN values in type I have to fill them. For now, we will not touch the column 'neighbourhood'

for column in ['type']:

     dataset[column].fillna(dataset[column].value_counts().index[0], inplace=True)

    
dataset.info()
#using Plotly

fig = px.histogram(dataset, x="type",title="Type of housing")

fig.show()
fig = px.histogram(dataset, x="price", color="type", marginal="violin", 

                         hover_data=dataset.columns, title="Price distribution according to type")

fig.show()
dataset['type'][dataset.description == 'Chalet en Puerta de la Carne - Judería, Sevilla'] = "Chalet"
dataset['type'][dataset.description == 'Chalet adosado en calle Palacios Malaver, Feria, Sevilla'] = 'Chalet_adosado'
#plot again and fixed

fig = px.histogram(dataset, x="price", color="type", marginal="violin", 

                         hover_data=dataset.columns, title="Price distribution according to type")

fig.show()
fig = px.histogram(dataset, x="neighbourhood",title="Neighbourhood with more houses to sale")

fig.show() #without filling the nan values 
dataset.zipcode.unique()
def f(row):

    if row['zipcode'] == 41001:

        val = 'Arenal-Museo'

    if row['zipcode'] ==41011:

        val = 'Tablada'

    if row['zipcode'] ==41018:

        val = 'La Buhaira'

    if row['zipcode'] ==41004:

        val = 'San Bernardo'

    if row['zipcode'] ==41004:

        val = 'San Bernardo'

    if row['zipcode'] ==41010:

        val = 'Triana/Los Remedios'

    if row['zipcode'] ==41003:

        val = 'Feria'

    if row['zipcode'] ==41009:

        val = 'Polígono Norte'

    if row['zipcode'] ==41013:

        val = 'La Palmera-Los Bermejales-Prado San Sebastián-Felipe II-Bueno Monreal'

    if row['zipcode'] ==41002:

        val = 'San Vicente-San Lorenzo-San Gil-Alameda'

    if row['zipcode'] ==41014:

        val = 'Bellavista'

    if row['zipcode'] ==41012:

        val = 'Heliópolis'

    if row['zipcode'] ==41005:

        val = 'Nervión'

    if row['zipcode'] ==41008:

        val = 'La Rosaleda'

    if row['zipcode'] ==41020:

        val = 'Valdezorras'

    if row['zipcode'] ==41015:

        val = 'Sevilla Norte'

    if row['zipcode'] ==41007:

        val = 'San Pablo-Santa Clara'

    if row['zipcode'] ==41006:

        val = 'Cerro-Amate'

    if row['zipcode'] ==41016:

        val = 'Torreblanca'

    if row['zipcode'] ==41019:

        val = 'Sevilla Este'

    return val
dataset['District'] = dataset.apply(f, axis=1)
dataset.head()
dataset.info()
fig = px.histogram(dataset, x="District",title="Houses on sale by districts")

fig.show() 