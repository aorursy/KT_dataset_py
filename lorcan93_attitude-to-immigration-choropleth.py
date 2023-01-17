import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import geopandas as gpd

from shapely import wkt

from shapely.geometry import Point, Polygon



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



##To graphically represent the number of refugees in EU countries per capita using scaled circles over country and attitude

#towards immigration via colour spectrum on geographical area of country via choropleth graph



#list of EU countries to extract from refugee destination dataframe

euCountries = ['Austria', 'Belgium', 'Bulgaria','Bosnia and Herzegovina', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'United Kingdom', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Montenegro', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Spain', 'Sweden', 'Switzerland']



##load refugee destination dataset from UNHCR

refPop = pd.read_csv('/kaggle/input/data-for-project/refugee_population_by_destination.csv', skiprows=4, usecols=['Country Name', '2018']) 

refPop.fillna(0)

refPop.isna().sum()



refPop.set_index('Country Name')

euRef = refPop[refPop['Country Name'].isin(euCountries)] #filter out countries not in EU



#load dataset of EU attitudes to immigration per 'Eurobarometer, March 2018' via europarl.europa.eu

refAttitude = pd.read_csv('/kaggle/input/data-for-project/attitudes_to_migrants.csv', header=0)

refAttitude['Country'].replace({'Great Britain': 'United Kingdom', 'czechia': 'Czech Republic'}, inplace=True)



#load countries coordinates for circles

countryCoor = pd.read_csv('/kaggle/input/countrylatlon/countries-latlon-csv-(Crawl-Run)---2019-11-11T180710Z.csv')

euCoor = countryCoor[countryCoor['Name'].isin(euCountries)]

euCoor = euCoor[['Name', 'Country', 'Latitude', 'Longitude', 'url']]

euCoor.set_index('Name')

geometry = [Point(xy) for xy in zip(euCoor['Longitude'], euCoor['Latitude'])]

crs = {'init': 'epsg:3035'}

euCoor = gpd.GeoDataFrame(euCoor, crs=crs, geometry=geometry)



#load europe map shapefile and change projection to more visible scale

euMapFile = '/kaggle/input/data-for-project/MyEurope.shp'

euMap = gpd.read_file(euMapFile)

euMap = euMap.to_crs({'init' : 'epsg:3035'})



#combine datasets, adding refugee population column, and surveyed attitude toward immigration from other datasets

joinedMap = euMap.set_index('SOVEREIGN').join(refAttitude.set_index('Country')).join(euRef.set_index('Country Name'))

refuPoints = euCoor.set_index('Name').join(euRef.set_index(['Country Name']),how='left', lsuffix='_left', rsuffix='_right')



#set variable for column to be visualised 

variable = 'Rating'

#set the range for the choropleth

vmin, vmax = 0, 100

# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(40, 24))



#plot map and colour spectrum per variable, no data displayed as grey

joinedMap.plot(ax=ax, color="grey", linewidth=1.0, edgecolor='0.8')

joinedMap.dropna().plot(ax=ax, column=variable, cmap='OrRd')

#refuPoints.plot(ax=ax, markersize = 1000, color = 'white', marker = 'o')



ax.axis('off')



# add title

ax.set_title('Positive Attitudes to Immigration (%)', fontdict={'fontsize': '32', 'fontweight' : '3'}, horizontalalignment='right')

# create an annotation for the data source

ax.annotate('Source: Eurobarometer, March 2018', xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='right', verticalalignment='top', fontsize=18, color='#555555')



# Create colorbar as a legend

sm = plt.cm.ScalarMappable(cmap='OrRd', norm = plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm._A = []

# add the colorbar to the figure

cbar = fig.colorbar(sm)


