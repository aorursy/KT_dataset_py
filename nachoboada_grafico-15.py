# Start with loading all necessary libraries

import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

%matplotlib inline
import geopandas as gpd
#import warnings

#warnings.filterwarnings("ignore")
# Load in the dataframe

df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)

df_2 = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)
df.head(2).T

#df.info()
df.describe(include='all')
df.info()
df2=df.dropna(subset=['provincia'])

df2.nunique()
df2.provincia.value_counts().to_list()
df = pd.DataFrame({'City': ['Distrito Federal', 'Edo. de México', 'Jalisco', 'Querétaro',

       'Nuevo León', 'Puebla', 'San luis Potosí', 'Yucatán', 'Morelos',

       'Veracruz', 'Quintana Roo', 'Chihuahua', 'Coahuila',

       'Baja California Norte', 'Sonora', 'Guanajuato', 'Guerrero', 'Hidalgo',

       'Michoacán', 'Tamaulipas', 'Durango', 'Sinaloa', 'Aguascalientes',

       'Baja California Sur', 'Nayarit', 'Chiapas', 'Colima', 'Tabasco',

       'Tlaxcala', 'Oaxaca', 'Campeche', 'Zacatecas'],

'Country': ['Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico', 'Mexico'],

     'Latitude': [19.49,19.35,20.62,20.58,25.67,18.46,22.00,20.97,18.88,19.18,21.17,28.63,25.54,31.87,29.33,21.12,16.84,20.11,19.70,25.87,24.02,26.37,21.88,24.14,21.50,16.75,15.67,17.98,19.31,17.06,19.84,22.76],

     'Longitude': [-99.12,-99.63,-103.23,-100.38,-100.45,-97.39,-99.01,-89.61,-99.17,-96.14,-86.84,-106.08,-103.41,-116.60,-110.66,-101.67,-99.90,-98.73,-101.18,-97.50,-104.65,-108.98,-102.28,-110.30,-104.89,-93.11,-93.27,-92.93,-98.19,-96.72,-90.52,-102.58],

     'Cantidad':[58790, 41607, 21238, 16988, 15324, 10421, 8447, 7928, 7337, 5762, 4756, 4590, 3695, 3220, 2988, 2860, 2678, 2521, 2471, 2303, 2275, 1806, 1753, 1700, 1352, 1126, 1008, 994, 839, 711, 263, 94]})
df.head()
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

type(gdf)
print(gdf)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



ax = world[world.iso_a3 == 'MEX'].plot(color='white', edgecolor='black')



gdf.plot(markersize=gdf['Cantidad']/100,ax=ax, color='blue',alpha=0.55)

gdf.plot(markersize=gdf['Cantidad']/10000,ax=ax, color='red')

ax.title.set_text("Distribucion de las Cantidad de Propiedades por Provincia")

ax.set_ylabel('Latitud')

ax.set_xlabel('Longitud')

plt.show()
df2=df[['City','Cantidad']].head()

ax= df2.plot(kind='bar',x='City', y='Cantidad',label='Cantidad por Ciudad')

ax.set_ylabel('Cantidad Avisos por Ciudad')

ax.set_xlabel('Ciudades')

#ax.legend()#.remove()

ax.title.set_text("Top 5 Cantidad de Propiedades por Ciudad");

df3=df[['City','Cantidad']].tail()

ax= df3.plot(kind='bar',x='City', y='Cantidad',label='Cantidad por Ciudad')

ax.set_ylabel('Cantidad Avisos por Ciudad')

ax.set_xlabel('Ciudades')

#ax.legend()#.remove()

ax.title.set_text("Tail 5 Cantidad de Propiedades por Ciudad");