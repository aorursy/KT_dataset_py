import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

import geopandas as gpd

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)
#Porcentaje de datos no nulos en cuanto a coordenadas



(df[['lat','lng']].dropna(how='all').shape)[0]/(df[['lat','lng']].shape)[0]
df['lat'] = df['lat'].dropna()

df['lng'] = df['lng'].dropna()

df['provincia'] = df['provincia'].dropna()

d = df
d = d.groupby('provincia')

d = d.agg({'lat':'mean','lng':'mean'})

d.columns = ['provlat','provlng']
df = pd.merge(d,df,on='provincia')
d = df.groupby('provincia')

d = round(d.agg({'antiguedad':'mean'})).astype(int)

d.columns = ['antiguedad promedio en provincia']
df = pd.merge(d,df,on='provincia')
#mapa mexico provincias en punto donde el color es que categoria de provincia es en cuanto a 

#magnitud en cantidad de propiedades"



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_17 = sns.scatterplot(data = df, y = 'provlat', x = 'provlng', size = 'antiguedad promedio en provincia'

                       , alpha = 0.05, sizes = (500, 12000), palette = 'spring')



plt.title("Provincias con tamaño segun la antiguedad promedio de sus propiedades")

plt.xlabel('longitud')

plt.ylabel('latitud')