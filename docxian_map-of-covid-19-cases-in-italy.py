import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import folium



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/covid-19-italy-updated-regularly/regional_data.csv')
max_date = df.data.max()

print(max_date)
# select most recent data

df_select = df[df.data==max_date]

df_select
n_regions = df_select.shape[0]

print('Number of regions: ', n_regions)
lat = 42

lon = 14

m = folium.Map([lat, lon], zoom_start=6)

for i in range(0,n_regions):

   folium.Circle(

      location=[df_select.lat.iloc[i], df_select.long.iloc[i]],

      popup=df_select.denominazione_regione.iloc[i]+'; cases: '+str(df_select.totale_casi.iloc[i]),

      radius=200*np.sqrt(df_select.totale_casi.iloc[i]),

      color='red',

      fill=True,

      fill_color='red'

   ).add_to(m)



# show map

m
df_prov = pd.read_csv('/kaggle/input/covid-19-italy-updated-regularly/provincial_data.csv')

df_prov = df_prov[df_prov.denominazione_provincia!='In fase di definizione/aggiornamento']
max_date = df_prov.data.max()

print(max_date)
# select most recent data

df_select_prov = df_prov[df_prov.data==max_date]

df_select_prov
n_provs = df_select_prov.shape[0]

print('Number of provinces: ', n_provs)
lat = 42

lon = 14

m = folium.Map([lat, lon], zoom_start=6)

for i in range(0,n_provs):

   folium.Circle(

      location=[df_select_prov.lat.iloc[i], df_select_prov.long.iloc[i]],

      popup=df_select_prov.denominazione_provincia.iloc[i]+'; cases: '+str(df_select_prov.totale_casi.iloc[i]),

      radius=200.0*np.sqrt(df_select_prov.totale_casi.iloc[i]),

      color='red',

      fill=True,

      fill_color='red'

   ).add_to(m)



# show map

m