# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import plotly.express as px


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
houses_v2 = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
houses_v2.head(5)
housesByCity = houses_v2.groupby(['city'])['city'].count()
housesByCity = housesByCity.to_frame('count').reset_index().sort_values(by='count', ascending=False)

fig = px.bar(housesByCity, x='city', y='count', text='count', width=600, height=400)
fig.update_layout(
    height=400,
    width = 600,
    title_text='Quantidade de aluguel por cidades',
    title_x=0.5)
fig.show()
#Média de valores aluguel

meanRentCity= houses_v2.groupby(['city'])['total (R$)'].mean().round(2)
meanRentCity = meanRentCity.to_frame('avgRent').reset_index().sort_values(by='avgRent', ascending=False)

fig = px.bar(meanRentCity, x='city', y='avgRent', color='city', text='avgRent')
fig.update_traces(textposition='inside')

fig.update_layout(
    height=400,
    width = 600,
    title_text='Média de valores de aluguel por cidades',
    title_x=0.5)
fig.show()
housesAcceptAnimal = houses_v2.groupby(['city','animal'])['city'].count()
housesAcceptAnimal = housesAcceptAnimal.to_frame('count').reset_index()

fig = px.scatter(housesAcceptAnimal, x="city", y="count", color="city", size='count', width=600, height=400)
fig.update_layout(
    height=400,
    width = 600,
    title_text='PETS POR CIDADE',
    title_x=0.5)

fig.show()
qtd_rooms = houses_v2.groupby(['city','rooms'])['rooms'].count().to_frame('count').reset_index().sort_values(by=['count','rooms'], ascending=False)
qtd_rooms

fig = px.bar(qtd_rooms, x="city", y="rooms", color="count", width=600, height=400)
fig.show()




