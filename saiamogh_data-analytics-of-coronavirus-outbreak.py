# Importing the packages required



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Geopandas

import geopandas  as gpd

from shapely.geometry import Point,Polygon

import descartes

# Time Series

import datetime as dt
df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
df.head()
df.columns
df.rename(columns={'Province/State':'Province_State','Country/Region':'Country_Region'},inplace = True)
# Changed Column Names

df.columns
# Shape of the Dataset

df.shape
# Data Types

df.dtypes
# Check the first 10 

df.head(10)
# Checking Missing Values



df.isnull().sum()
df.describe()
# Number of Cases Per Day

df.head()
df.groupby('Date').agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum'}).head()
# Per day max

df_per_day= df.groupby('Date')['Confirmed','Deaths', 'Recovered'].max()

df_per_day.describe()


print('Minimum Confirmed Cases On a single day {}'.format(df_per_day['Confirmed'].min()))

print('Max Confirmed Cases On a single day {}'.format(df_per_day['Confirmed'].max()))
# Date For Maximum number of cases

df_per_day['Confirmed'].idxmax()
# Date For Minimum  number of cases

df_per_day['Confirmed'].idxmin()
# Number Of Cases Per Country/Province



df.groupby(['Country_Region'])['Confirmed','Deaths', 'Recovered'].max().reset_index().sort_values('Confirmed',ascending  = False)
# Number Of Cases Per Country/Province



df.groupby(['Country_Region','Province_State'])['Confirmed','Deaths', 'Recovered'].max().reset_index().sort_values('Confirmed',ascending  = False)
df['Country_Region'].value_counts().head()
df['Country_Region'].value_counts().plot(kind = 'bar',figsize = (20,10))
# How Many Countries 

df['Country_Region'].unique()
len(df['Country_Region'].unique())
dir(gpd)
# Convert data to geodataframe



gdf01 = gpd.GeoDataFrame(df,geometry = gpd.points_from_xy(df['Long'],df['Lat']))
gdf01.head()
type(gdf01)
# Method 2

points = [Point(x,y) for x,y in zip(df['Long'],df['Lat'])]
gdf02 = gpd.GeoDataFrame(df,geometry = points)
# Map Plot 



gdf02.plot(figsize =(20,10))
# Overlap with World Map



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

ax = world.plot(figsize = (20,10))

ax.axis('off')
# Overlap



fig,ax = plt.subplots(figsize = (20,10))

gdf01.plot(cmap = 'Purples',ax=ax)

world.geometry.boundary.plot(color=None,edgecolor = 'k',linewidth=2,ax = ax)
#Per Country 



world['continent'].unique()
Asia = world[world['continent'] == 'Asia']

Africa = world[world['continent'] == 'Africa']

North_america = world[world['continent'] == 'North America']

Europe = world[world['continent'] == 'Europe']
# Overlap



fig,ax = plt.subplots(figsize = (20,10))

gdf01[gdf01['Country_Region'] == 'Mainland China'].plot(cmap = 'Purples',ax=ax)

Asia.geometry.boundary.plot(color=None,edgecolor = 'k',linewidth=2,ax = ax)
# Time Series Analytics



df.head()

df_per_day.head()
# Copy

df2 = df
df['cases_date'] = pd.to_datetime(df2['Date'])
df.dtypes
df['cases_date'].plot()
ts = df2.set_index('cases_date')
ts.loc['2020-01']
ts.loc['2020-01-24' :'2020-02-25'][['Confirmed','Recovered']].plot(kind = 'line',figsize= (20,10))
df_date = ts.groupby(['cases_date']).sum().reset_index(drop =None)
df_date[['Confirmed','Recovered','Deaths']].plot(kind = 'line',figsize = (20,10))