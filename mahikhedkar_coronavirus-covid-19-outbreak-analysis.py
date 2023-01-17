#Load EDA pkgs
import pandas as pd
import numpy as np
#Load Data Visualization pkgs
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Load GeoPandas
import geopandas as gpd
from shapely.geometry import Point, Polygon
import descartes
#Load Dataset
data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
data.head()
data.columns
data.rename(columns={'Province/State':'Province_State','Country/Region':'Country_Region'},inplace=True)
data.columns
#shape of data
data.shape
# datatypes
data.dtypes
# First 10
data.head(10)
data=data[['Province_State', 'Country_Region', 'Lat', 'Long', 'Date', 'Confirmed', 'Deaths', 'Recovered' ]]
data.isna().sum()
data.describe()
# Number of case per Date/Day
data.head()
data.groupby('Date')['Confirmed','Deaths','Recovered'].sum()
data.groupby('Date')['Confirmed','Deaths','Recovered'].max()
data_per_day = data.groupby('Date')['Confirmed','Deaths','Recovered'].max()
data_per_day.head()

data_per_day.describe()
# Max No of cases
data_per_day['Confirmed'].max()
# Min No of cases
data_per_day['Confirmed'].min()
# Date for Maximum Number Cases 
data_per_day['Confirmed'].idxmax()
# Date for Minimun Number Cases
data_per_day['Confirmed'].idxmin()
#Number of Case Per Country/Province
data.groupby(['Country_Region'])['Confirmed','Deaths','Recovered'].max()
# Number of Case Per Country/Province
data.groupby(['Province_State','Country_Region'])['Confirmed','Deaths','Recovered'].max()
data['Country_Region'].value_counts()
data['Country_Region'].value_counts().plot(kind='bar',figsize=(30,10))
#How Many Country Affect
data['Country_Region'].unique()
# How Many Country Affect
len(data['Country_Region'].unique())
plt.figure(figsize=(30,30))
data['Country_Region'].value_counts().plot.pie(autopct="%1.1f%%")

dir(gpd)
data.head()
# Convert Data to GeoDataframe
gdata01 = gpd.GeoDataFrame(data,geometry=gpd.points_from_xy(data['Long'],data['Lat']))
gdata01.head()
type(gdata01)
# Method 2
points = [Point(x,y) for x,y in zip(data.Long,data.Lat)]
gdata02 = gpd.GeoDataFrame(data,geometry=points)
gdata02
#Map Plot
gdata01.plot(figsize=(20,10))
# Overlapping With World Map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(20,10))
ax.axis('off')
# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01.plot(cmap='Purples',ax=ax)
world.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)
fig,ax = plt.subplots(figsize=(20,10))
gdata01.plot(cmap='Purples',ax=ax)
world.geometry.plot(color='Yellow',edgecolor='k',linewidth=2,ax=ax)
world['continent'].unique()
asia = world[world['continent'] == 'Asia']
asia
africa = world[world['continent'] == 'Africa']
north_america = world[world['continent'] == 'North America']
europe = world[world['continent'] == 'Europe']
# Cases in China
data.head()
data[data['Country_Region'] == 'China']

gdata01[gdata01['Country_Region'] == 'China']

# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'China'].plot(cmap='Purples',ax=ax)
world.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)
# Overlaph
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'China'].plot(cmap='Purples' ,ax=ax)
asia.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)
#Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region']== 'India'].plot(cmap='Purples',ax=ax)
asia.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)
# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'Egypt'].plot(cmap='Purples',ax=ax)
africa.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)
# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'US'].plot(cmap='Purples',ax=ax)
north_america.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)
# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'United Kingdom'].plot(cmap='Purples',ax=ax)
europe.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)
data.head()
data_per_day
data2 = data
data.to_csv("E:\covid_19_clean_complete.csv")
import datetime as dt
data['cases_date'] = pd.to_datetime(data2['Date'])
data2.dtypes
data['cases_date'].plot(figsize=(20,10))
ts = data2.set_index('cases_date')
# Select For January
ts.loc['2020-01']
# Select For January
ts.loc['2020-01']
ts.loc['2020-02-24':'2020-02-25']
ts.loc['2020-02-24':'2020-02-25']
ts.loc['2020-02-24':'2020-02-25'][['Confirmed','Recovered']].plot(figsize=(20,10))
ts.loc['2020-02-2':'2020-02-25'][['Confirmed','Deaths']].plot(figsize=(20,10))
data_by_date = ts.groupby(['cases_date']).sum().reset_index(drop=None)
data_by_date
data_by_date.columns
data_by_date[['Confirmed', 'Deaths', 'Recovered']].plot(kind='line',figsize=(20,10))