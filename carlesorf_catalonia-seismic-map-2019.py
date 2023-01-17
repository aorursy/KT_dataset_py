import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns

import folium

import warnings

warnings.filterwarnings('ignore')
df = pd.read_excel('/kaggle/input/aec2019/aec-209.xls')
df.head()
df.dtypes
df.LN = df.LN.str.replace(' N', '')

df.LN = df.LN.str.replace(',', '.')

df.LE = df.LE.str.replace(' E', '')

df.LE = df.LE.str.replace(',', '.')

df['focuskm'] = df['focuskm'].astype(float)





for idx, row in df.iterrows():

    if 'W' in row['LE']:

        df['LE'][idx] = df['LE'][idx].replace(' W', '')

        df['LE'][idx] = float(df['LE'][idx])

        df['LE'][idx] = -df['LE'][idx]

for key in df.index:

    #print(key)

    folium.Marker([df['LN'][key], df['LE'][key]], popup=df['Regio'][key]).add_to(cat_map)

cat_map
cat_map2 = folium.Map(

    location=[41.540611, 2.114665],

    #tiles='Stamen Terrain',

    zoom_start=8)



for key in df.index:

   folium.Circle(

      location=[df['LN'][key], df['LE'][key]],

      popup=df['Regio'][key],

      radius=df['mag'][key]*3000,

      color='crimson',

      fill=True,

      fill_color='crimson'

   ).add_to(cat_map2)



cat_map2
cat_map3 = folium.Map(

    location=[41.540611, 2.114665],

    #tiles='Stamen Terrain',

    zoom_start=8)



for key in df.index:

   folium.Circle(

      location=[df['LN'][key], df['LE'][key]],

      popup=df['Regio'][key],

      radius=df['focuskm'][key]*1000,

      color='crimson',

      fill=True,

      fill_color='crimson'

   ).add_to(cat_map3)



cat_map3