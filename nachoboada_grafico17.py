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
def crearDiccionarioMapeandoProvinciaConSuCategoria(df):

    

    dic = {}

    for provincia in df['provincia']:

        

        if not (provincia in dic):

            n = np.log10((df['provincia'] == provincia).sum() + 1).astype(int)

            dic[provincia] = n

        

    return dic
dic = crearDiccionarioMapeandoProvinciaConSuCategoria(df)
df['orden de magnitud en cantidad de propiedades'] = df['provincia'].map(dic)
#mapa mexico provincias en punto donde el color es que categoria de provincia es en cuanto a 

#magnitud en cantidad de propiedades"



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_17 = sns.scatterplot(data = df, y = 'provlat', x = 'provlng', hue = 'orden de magnitud en cantidad de propiedades',

                       size = 'orden de magnitud en cantidad de propiedades', alpha = 0.8, palette = 'tab10',

                       sizes = (1500,5000))



plt.title("Provincias con color segun la magnitud de propiedades que tienen")

plt.xlabel('longitud')

plt.ylabel('latitud')