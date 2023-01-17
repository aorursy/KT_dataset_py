# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





import geopandas as gpd

from shapely.geometry import LineString

from geopandas.tools import geocode

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, FastMarkerCluster

from folium import plugins

import math

import webbrowser

from IPython.display import HTML

import matplotlib.pyplot as plt

from pandasql import sqldf

import plotly.express as px



#turn off settingwithcopywarning off

pd.options.mode.chained_assignment = None
#Have a look at initial dataset

US_covid_data = pd.read_csv("../input/us-counties-covid-19-dataset/us-counties.csv")

#Restrict dates to be before 11th April

US_covid_data = US_covid_data.loc[US_covid_data['date']<'2020-04-11']

#Create a concatenated column - not actually important since we'll map with fips column

US_covid_data['concat'] = US_covid_data['county']+str(', ')+US_covid_data['state']



US_covid_data = US_covid_data.sort_values(['date'], ascending = True)

US_covid_data
US_covid_data['concat'] = US_covid_data['county']+str(', ')+US_covid_data['state']

US_covid_data = US_covid_data.sort_values(['date'], ascending = True)

US_covid_data
print("Values in fips column prior to manipulation: \n" +str(US_covid_data.loc[US_covid_data['county'] == 'New York City'].fips.value_counts())+"\n")

US_covid_data.loc[US_covid_data['county'] == 'New York City'] = US_covid_data.loc[US_covid_data['county'] == 'New York City'].fillna(36061.0)

print("Values in fips column after manipulation: \n" +str(US_covid_data.loc[US_covid_data['county'] == 'New York City'].fips.value_counts()))
q1='select DISTINCT date FROM US_covid_data'

df_new=sqldf(q1)

df_new['day'] = np.arange(len(df_new))

df_new
concat_result = US_covid_data.merge(df_new,on='date', how = 'left')

concat_result.head()
us_counties_shapefile = gpd.read_file("../input/us-counties-geocoded/tl_2017_us_county.shp")

us_counties_shapefile.head()

us_counties_dataframe = pd.DataFrame(us_counties_shapefile[['GEOID', 'INTPTLAT', 'INTPTLON']])

us_counties_dataframe['GEOID'] = us_counties_dataframe['GEOID'].astype('float64')



concat_result2 = concat_result.merge(us_counties_dataframe,left_on = 'fips', right_on = 'GEOID', how = 'left')



concat_result2.head()
#Fill in 'fips' column with arbitrary number = 1

concat_result2.loc[concat_result2['county'] == 'Kansas City','fips'] = concat_result2.loc[concat_result2['county'] == 'Kansas City','fips'].fillna(1)



#Fill in 'GEOID' column with arbitrary number = 1

concat_result2.loc[concat_result2['county'] == 'Kansas City','GEOID'] = concat_result2.loc[concat_result2['county'] == 'Kansas City','GEOID'].fillna(1)



#Fill in 'INTPTLAT' column with +39.09970

concat_result2.loc[concat_result2['county'] == 'Kansas City','INTPTLAT'] = concat_result2.loc[concat_result2['county'] == 'Kansas City','INTPTLAT'].fillna('+39.0997000')



#Fill in 'INTPTLON' column with -94.57860

concat_result2.loc[concat_result2['county'] == 'Kansas City','INTPTLON'] = concat_result2.loc[concat_result2['county'] == 'Kansas City','INTPTLON'].fillna('-94.5786000')

concat_result2['INTPTLAT'] = concat_result2['INTPTLAT'].astype('str')

concat_result2['INTPTLAT']

concat_result2['INTPTLAT'] = concat_result2['INTPTLAT'].str[1:]

concat_result2['INTPTLAT']



concat_result2.dropna(how = 'any', inplace = True)



concat_result_cases = concat_result2.loc[concat_result2.index.repeat(concat_result2['cases'])]

print("Shape of dataset to be mapped: " +str(concat_result_cases.shape))





concat_result_cases['INTPTLAT'] = concat_result_cases['INTPTLAT'].astype('float64')

concat_result_cases['INTPTLON'] = concat_result_cases['INTPTLON'].astype('float64')
max_date = concat_result_cases['day'].max()



heat_data = [[[row['INTPTLAT'],row['INTPTLON']] 

              for index, row in concat_result_cases[concat_result_cases['day'] == i].iterrows()] for i in range(0,max_date)]



map4 = folium.Map(location=[40, -95], zoom_start=4)



hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8, min_speed = 4, overlay=False,radius = 16.5, display_index=True)

hm.add_to(map4)



map4.save('plot_data4.html')   

HTML('<iframe src=plot_data4.html width=800 height=600></iframe>')

        