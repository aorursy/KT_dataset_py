import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import geopandas as gpd

from shapely.geometry import Polygon , Point

import descartes
df = pd.read_csv("../input/covid19-info/coronavirus_data.csv")

df.head()

df.columns
df.columns = df.columns.str.replace(r'\n','',regex=True)

df.rename(columns={'Province/State':'Province_State','Country/Region':'Country_Region'}, inplace = True)

df
df.shape
df.dtypes
df.columns
df = df[['Province_State', 'Country_Region', 'Lat', 'Long', 'Date',

       'Confirmed', 'Deaths', 'Recovered']]

df.describe()
#deaths and data per day



df.groupby('Date')['Confirmed','Deaths','Recovered'].sum().head()
df.groupby('Date')['Confirmed','Deaths','Recovered'].max().head()
df_per_day = df.groupby('Date')['Confirmed','Deaths','Recovered'].max()
df_per_day.head()
df_per_day.describe()
#max cases 

df_per_day['Confirmed'].max()
#Date for max no cases 

df_per_day['Confirmed'].idxmax()
#no of cases per country 

df.groupby('Country_Region')['Confirmed','Deaths','Recovered'].max()
#no of cases per country 

df.groupby(['Province_State','Country_Region'])['Confirmed','Deaths','Recovered'].max()
df['Country_Region'].value_counts().plot(kind='bar',figsize=(20,10))
df['Country_Region'].unique()
# distribution on map

dir(gpd)
#converting data to geo dataframe

gdf1 = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df['Long'],df['Lat']))
gdf1.head()
#map plot

gdf1.plot(figsize=(20,10))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

ax = world.plot(figsize=(20,10))

ax.axis('off')
#overlapping 

fig,ax = plt.subplots(figsize=(20,10))

gdf1.plot(cmap='Blues',ax=ax)

world.geometry.boundary.plot(color=None,edgecolor= 'c',linewidth=1,ax=ax)
world
asia = world[world['continent']=="Asia"]

gdf1
#overlapping 

fig,ax = plt.subplots(figsize=(20,10))

gdf1[gdf1['Country_Region']=='Mainland China'].plot(cmap='Blues',ax=ax)

asia.geometry.boundary.plot(color=None,edgecolor= 'k',linewidth=2,ax=ax)