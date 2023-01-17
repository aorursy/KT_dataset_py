import numpy as np

import pandas as pd



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



import plotly.express as px

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
#reading in training data and test data

train_data = pd.read_csv("/kaggle/input/sf-crime/train.csv.zip")
#Subsetting Data for Just Juvenile Crime Cases

array_juv = ['JUVENILE BOOKED','JUVENILE CITED','JUVENILE ADMONISHED','JUVENILE DIVERTED']

juvenile_crime = train_data.loc[train_data['Resolution'].isin(array_juv)]
#Getting Rid of Outliers who's x or y coordinates are not in San Francisco

juvenile_crime = juvenile_crime[juvenile_crime['Y'] < 60]
#changing type of dates to DateTime

juvenile_crime['Dates'] = pd.to_datetime(juvenile_crime['Dates'])



#Creating new columns for month, year, and time_of_day

juvenile_crime['year'] = pd.DatetimeIndex(juvenile_crime['Dates']).year

juvenile_crime['month'] = pd.DatetimeIndex(juvenile_crime['Dates']).month

juvenile_crime['time_of_day'] = pd.DatetimeIndex(juvenile_crime['Dates']).hour
#Creating bar graph: count of each police district

juvenile_crime['month'].value_counts()[[1,2,3,4,5,6,7,8,9,10,11,12]].plot(kind = 'bar', figsize = (10,6), color = 'lightblue', edgecolor = 'black',title = 'Juvenile Crime Count by Month')
juvenile_crime['DayOfWeek'].value_counts()[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']].plot(kind = 'bar', figsize = (10,6), color = 'lightblue', edgecolor = 'black',title = 'Juvenile Crime Count by Day of Week')
juvenile_crime['time_of_day'].value_counts()[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]].plot(kind = 'bar', figsize = (10,6), color = 'lightblue', edgecolor = 'black',title = 'Juvenile Crime Count by Hour')
#Creating bar graph: count of each Category of Crime(only plotted 10 most common)

juvenile_crime['Category'].value_counts().head(15).plot(kind = 'bar', figsize = (11,6), color = 'lightblue', edgecolor = 'black',title = 'Juvenile Crime Count by Category')
#Creating bar graph: count of each description of crime (only plotted 10 most common)

juvenile_crime['Descript'].value_counts().head(15).plot(kind = 'bar', figsize = (11,6), color = 'lightblue', edgecolor = 'black', title = 'Juvenile Crime Count by Description')
juvenile_crime['PdDistrict'].value_counts().plot(kind = 'bar', figsize = (10,6), color = 'lightblue', edgecolor = 'black',title = 'Juvenile Crime Count by Police District')
from shapely.geometry import Point

import geopandas as gpd

import folium

from folium.plugins import MarkerCluster

from folium.plugins import FastMarkerCluster
#reading in data to creat boundaries for the police disctrics

#taken from https://data.sfgov.org/Public-Safety/Historical-Police-Districts/embj-38bg

police_districts_before_2015 = gpd.read_file('/kaggle/input/historical-police-districts/Historical Police Districts.geojson')
#creating geometry for my juvenile crime from x and y coordinates

juvenile_crime['geometry'] = juvenile_crime.apply(lambda q: Point((float(q.X), float(q.Y))), axis=1)

juvenile_crime_geo = gpd.GeoDataFrame(juvenile_crime, geometry = juvenile_crime['geometry'])
def make_crime_cat_plot(category):

    category_map_data = juvenile_crime_geo.loc[juvenile_crime['Category']== category]

    

    leg_kwds = {'title': 'Police District', 'loc': 'upper left','bbox_to_anchor': (1, 1.03)}



    ax = police_districts_before_2015.plot(column = 'district', label = 'district', figsize=(10, 10),

                                          edgecolor = 'black', legend = True, legend_kwds = leg_kwds, 

                                          cmap = 'Pastel1')



    category_map_data.plot(ax = ax, color = 'black');

    plt.title(category, weight='bold', size='large')

    plt.show();
make_crime_cat_plot('ASSAULT')

make_crime_cat_plot('DRUG/NARCOTIC')

make_crime_cat_plot('LARCENY/THEFT')

make_crime_cat_plot('ROBBERY')

make_crime_cat_plot('VANDALISM')
def make_crime_year_plot(year):

    year_map_data = juvenile_crime_geo.loc[juvenile_crime['year']== year]

    

    leg_kwds = {'title': 'Police District', 'loc': 'upper left','bbox_to_anchor': (1, 1.03)}



    ax = police_districts_before_2015.plot(column = 'district', label = 'district', figsize=(10, 10),

                                          edgecolor = 'black', legend = True, legend_kwds = leg_kwds, 

                                          cmap = 'Pastel1')



    year_map_data.plot(ax = ax, color = 'black');

    

    plt.title(year, weight='bold', size='large')

    

    plt.show();
make_crime_year_plot(2003)

make_crime_year_plot(2013)
def make_crime_time_plot(hour1,hour2, hour3):

    year_map_data = juvenile_crime_geo.loc[(juvenile_crime['time_of_day']== hour1)|(juvenile_crime['time_of_day']== hour2)|(juvenile_crime['time_of_day']== hour3)]

    

    leg_kwds = {'title': 'Police District', 'loc': 'upper left','bbox_to_anchor': (1, 1.03)}



    ax = police_districts_before_2015.plot(column = 'district', label = 'district', figsize=(10, 10),

                                          edgecolor = 'black', legend = True, legend_kwds = leg_kwds, 

                                          cmap = 'Pastel1')



    year_map_data.plot(ax = ax, color = 'black');

    

    plt.title(str(hour1) + ', '+ str(hour2) + ', '+ str(hour3) + ' Oclock', weight='bold', size='large')

    

    plt.show();
#afterschool

make_crime_time_plot(15,16,17)
#middle of night

make_crime_time_plot(22, 23, 1)