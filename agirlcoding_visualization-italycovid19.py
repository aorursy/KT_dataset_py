# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd 

from shapely.geometry import LineString

import geopandas as gpd

import geoplot

import seaborn as sns

import matplotlib.pyplot as plt

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

from matplotlib.pyplot import figure, show







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

Italy = world[world.name == 'Italy'].plot(figsize=(10,15), color='whitesmoke', linestyle=':', edgecolor='green')

world.head()
ItalyRegions=pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')

ItalyProvince=pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_province.csv')
GeoRegion = gpd.GeoDataFrame(ItalyRegions,geometry=gpd.points_from_xy(ItalyRegions['Latitude'],ItalyRegions['Longitude']))

GeoRegion.crs = {'init': 'epsg:4326'}

GeoRegion.head()

#GeoRegion.to_crs(epsg=4326).plot(markersize=2, ax=ax)

print(ItalyRegions.info())

GeoRegion.plot(color='lightgreen',markersize=2, ax=Italy)

GeoRegion
Startingmap = folium.Map(location=[42.351222,13.39844 ], tiles='cartodbpositron', zoom_start=7)

##for idx, row in GeoRegion.iterrows():

  #  Marker([row['Latitude'], row['Longitude']]).add_to(Startingmap)

#Startingmap

HeatMap(data=GeoRegion[['Latitude', 'Longitude']], radius=10).add_to(Startingmap)

Startingmap
symptoms=['Fever or chills','Cough','Shortness of breath or difficulty breathing','Fatigue','Muscle or body aches','Headache','New loss of taste or smell','Sore throat','Congestion or runny nose','Nausea or vomiting','Diarrhea']
from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in symptoms)

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
RegionsReduced=ItalyRegions[['Date','RegionName','Recovered','IntensiveCarePatients','HomeConfinement','CurrentPositiveCases','Deaths', 'TotalPositiveCases','TestsPerformed','NewPositiveCases']]

RegionsTotal = RegionsReduced.groupby('RegionName').sum().reset_index()

TimeChanges= RegionsReduced.groupby('Date').sum().reset_index()

print(TimeChanges.head())
#DeathbyRegion

figure(figsize=(20,6))

Deathbyregions=sns.barplot(data=RegionsTotal, x="RegionName", y="Deaths", order=RegionsTotal['RegionName'], palette='winter').set_title('Number of Death people due to Covid-19 per Region', fontweight='bold', color = 'teal', fontsize='18')

plt.xticks(rotation=90)

plt.xlabel('Region of Italy', fontweight='bold', color = 'darkturquoise', fontsize='14')

plt.ylabel('Number of people death due to COVID-19', fontweight='bold', color = 'darkturquoise', fontsize='14') 
figure(figsize=(20,6))

Deathbyregions=sns.barplot(data=RegionsTotal, x="RegionName", y="TotalPositiveCases", order=RegionsTotal['RegionName'], palette='winter').set_title('Number of Total Positive Cases per Region', fontweight='bold', color = 'teal', fontsize='18')

plt.xticks(rotation=90)

plt.xlabel('Region of Italy', fontweight='bold', color = 'darkturquoise', fontsize='14')

plt.ylabel('Number of Total Positive Cases', fontweight='bold', color = 'darkturquoise', fontsize='14')
figure(figsize=(20,6))

Deathbyregions=sns.barplot(data=RegionsTotal, x="RegionName", y="CurrentPositiveCases", order=RegionsTotal['RegionName'], palette='winter').set_title('Number of Current Positive Cases per Region', fontweight='bold', color = 'teal', fontsize='18')

plt.xticks(rotation=90)

plt.xlabel('Region of Italy', fontweight='bold', color = 'darkturquoise', fontsize='14')

plt.ylabel('Number of Current Positive Cases', fontweight='bold', color = 'darkturquoise', fontsize='14')
figure(figsize=(30,6))

plt.xticks(rotation=90)

time = sns.lineplot(x="Date", y="CurrentPositiveCases", data=TimeChanges).set_title('Number of Current Positive Cases per Day', fontweight='bold', color = 'teal', fontsize='18')

figure(figsize=(30,6))

plt.xticks(rotation=90)

time = sns.lineplot(x="Date", y="NewPositiveCases", data=TimeChanges).set_title('Number of New Positive Cases per Day', fontweight='bold', color = 'teal', fontsize='18')