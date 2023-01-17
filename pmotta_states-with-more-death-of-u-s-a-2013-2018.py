import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

%cd /kaggle/input/gun-violence-data
df = pd.read_csv('gun-violence-data_01-2013_03-2018.csv', usecols=['date', 'state', 'city_or_county', 'n_killed', 'n_injured', 'latitude', 'longitude', 'location_description'])



df.state = df.state.str.lower().str.strip()
plt.style.use('ggplot')



df.groupby('state').n_killed.sum().sort_values()[-20:].plot(kind='bar', color='darkred', figsize=(13,7))

plt.ylabel('Number of killed', fontsize=20)

plt.xlabel('State', fontsize=20)

plt.title('U.S.A', fontsize=25)
#DATA GRUPING FOR THE STATES WITH MORE DEATHS



california = df[df['state'] == 'california'].copy()

texas = df[df['state'] == 'texas'].copy()

florida = df[df['state'] == 'florida'].copy()
#FUNCTIONS



#DEATHS BY CITY



def mortos_por_cidade(state):



  state.groupby('city_or_county').n_killed.sum().sort_values()[-10:].plot(kind='bar', figsize=(15,5))

  plt.ylabel('Number of killed', fontsize=20)

  plt.xlabel('City or county', fontsize=20)



#DEATHS BY LOCATION



def locais_com_mais_atentados(state):



  state.groupby('location_description').n_killed.sum().sort_values()[-20:].plot(kind='bar', figsize=(15,5))

  plt.ylabel('Number of killed', fontsize=20)

  plt.xlabel('Location description', fontsize=20)

 
locais_com_mais_atentados(california)

plt.title('California', fontsize=25)
mortos_por_cidade(california)

plt.title('Citys of California', fontsize=25)
locais_com_mais_atentados(texas)

plt.title('Texas', fontsize=25)
mortos_por_cidade(texas)

plt.title('Citys of Texas', fontsize=25)
locais_com_mais_atentados(florida)

plt.title('Florida', fontsize=25)
mortos_por_cidade(florida)

plt.title('Citys of Florida', fontsize=25)
import folium

from folium import plugins
df.latitude.dropna(inplace=True)

df.longitude.dropna(inplace=True)
mapa = folium.Map(location=[40.788497,-79.879873], zoom_start=4)

coordenadas = []



for la,lo in zip( df.latitude[:], df.longitude[:]):

    

    coordenadas.append([la,lo])



mapa.add_child(plugins.HeatMap(coordenadas))



display(mapa)