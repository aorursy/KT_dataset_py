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
d1 = df[['lat','lng','provincia']].dropna()

d1 = d1.groupby('provincia')

d1 = d1.agg({'lat':'mean','lng':'mean'})
d1.head(20)
d2 = df[['centroscomercialescercanos','provincia']].dropna()

d2 = d2.groupby('provincia').agg({'centroscomercialescercanos':'sum'})

d2.columns = ['cantidad de centros comerciales']

d2 = d2.astype(int)
d2.sort_values(by = 'cantidad de centros comerciales', ascending=False).head(20)
d3 = pd.merge(d1,d2,on='provincia')

d3.head(20)
#mapa mexico provincias en punto donde el tamaño es cant de escuelas

gdf = gpd.GeoDataFrame(d3, geometry=gpd.points_from_xy(d3.lng, d3.lat))



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



ax = world[world.iso_a3 == 'MEX'].plot(color='white', edgecolor='black')



gdf.plot(markersize=gdf['cantidad de centros comerciales']/100,ax=ax, color='blue',alpha=0.55)

gdf.plot(markersize=gdf['cantidad de centros comerciales']/10000,ax=ax, color='red')

ax.title.set_text("Distribucion de las Cantidad de Centros Comerciales por Provincia")

ax.set_ylabel('Latitud')

ax.set_xlabel('Longitud')

plt.show()