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
density =  pd.read_csv("../input/us-counties-covid-19-dataset/us-counties.csv", dtype={"fips": str})

density = density.loc[density.date == density.date.max()]

print("NY City before update: \n" +str(density.loc[density['county'] == 'New York City'])+"\n")

density.loc[density['county'] == 'New York City','fips'] = 36061

print("NY City after update: \n" +str(density.loc[density['county'] == 'New York City'])+"\n")

#drop the rows with county = 'unknown' - these rows have no fips values

density.dropna(how = 'any', inplace = True)
unrecorded =  pd.read_csv('../input/us-counties-without-recorded-casesdeaths/unrecorded_counties.csv')



unrecorded_cases = unrecorded[['fips','county','cases']]

unrecorded_deaths = unrecorded[['fips','county','deaths']]



unrecorded_deathrate = unrecorded[['fips','county','cases','deaths']]

unrecorded_deathrate['death_rate'] = (unrecorded_deathrate['deaths']/unrecorded_deathrate['cases']).fillna(0)

unrecorded_deathrate = unrecorded_deathrate[['fips','county','death_rate']]
# Take what you need from initial US Covid-19 dataset 

density1 = density[['fips','county','cases']]

# Store what you need from initial US Covid-19 dataset and dataset of counties with no recorded cases as a frame

frames1 = [density1,unrecorded_cases]

# Concat the two of them together

concat_density1 = pd.concat(frames1)



#Same method as above but for deaths

density2 = density[['fips','county','deaths']]

frames2 = [density2,unrecorded_deaths]

concat_density2 = pd.concat(frames2)





#Same method as above but for death rate

density3 = density[['fips','county','cases','deaths']]

density3['death_rate'] = (density3['deaths']/density3['cases'])*100

density3 = density3[['fips','county','death_rate']]

frames3 = [density3,unrecorded_deathrate]

concat_density3 = pd.concat(frames3)
from urllib.request import urlopen

import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)

# Below will show that we should use feature.id as the way to link this dataset with our conca_density datasets

counties["features"][0]
concat_density1['cases'].describe()
fig1 = px.choropleth_mapbox(concat_density1, geojson=counties, locations='fips', color='cases',

                           color_continuous_scale="Viridis",

                            mapbox_style="carto-positron",

                           hover_name='county',

                           zoom=2.5, center = {"lat": 37.0902, "lon": -95.7129},

                           labels={'cases':'cases'}

                          )

fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig1.show()
fig2 = px.choropleth_mapbox(concat_density1, geojson=counties, locations='fips', color='cases',

                           color_continuous_scale="Viridis",

                           mapbox_style="carto-positron",

                           range_color=(0,1000),

                           hover_name='county',

                           zoom=2.5, center = {"lat": 37.0902, "lon": -95.7129},

                           labels={'cases':'cases'}

                          )

fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig2.show()
#Additional step required to assign each county a value based on the quartile that it is in

quartiles = (concat_density1['cases'].min(),

                 np.quantile(concat_density1['cases'], 0.25),

                 np.quantile(concat_density1['cases'], 0.5),

                 np.quantile(concat_density1['cases'], 0.75),

                 concat_density1['cases'].max())



def quantile_value(val):

    if quartiles[0] <= val < quartiles[1]:

        return '1'

    if quartiles[1] <= val < quartiles[2]:

        return '2'

    if quartiles[2] <= val < quartiles[3]:

        return '3'

    else:

        return '4'

    

concat_density1['quartile'] = concat_density1.apply(lambda x: quantile_value(x['cases']), axis=1)



#For whatever reason the choropleth_mapbox will assign colour values based on the order that the quartiles appear starting from row 1

#Without sorting, the map will assign colours based on the order 3,4,2,1

#By sorting from highest to lowest quartile values the map will now assign colours based on the order 4,3,2,1

concat_density1 = concat_density1.sort_values(by=['quartile'],ascending = False )

concat_density1.head()
#Only need 4 colours so print one of the plotly colour schemes to get the exact colour codes

print("Viridis colour codes")

print(px.colors.sequential.Viridis)
colours = ['#440154', '#31688e','#35b779','#fde725']



fig4 = px.choropleth_mapbox(concat_density1, geojson=counties, locations='fips', color='quartile',

                           mapbox_style="carto-positron",

                           hover_name='county',

                           color_discrete_sequence= colours,

                           zoom=2.5, center = {"lat": 37.0902, "lon": -95.7129},

                           labels={'cases':'cases'}

                          )

fig4.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig4.show()
concat_density2['deaths'].describe()
list = (np.arange(0.8,1.0,0.02))



for i in list:

    print("Percentile = " +str(i.round(2)) + ", Deaths = " +str(np.quantile(concat_density2['deaths'], i).round(0)))
fig5 = px.choropleth_mapbox(concat_density2, geojson=counties, locations='fips', color='deaths',

                           color_continuous_scale="Bluyl",

                           range_color=(0,127),

                            mapbox_style="carto-positron",

                           hover_name='county',

                           zoom=2.5, center = {"lat": 37.0902, "lon": -95.7129},

                           labels={'deaths':'deaths'}

                          )

fig5.update_layout(margin={"r":0,"t":0,"l":0,"b":0},title_text ='US Covid-19 Deathrate')

fig5.show()
concat_density3['death_rate'].describe()
fig6 = px.choropleth_mapbox(concat_density3, geojson=counties, locations='fips', color='death_rate',

                           color_continuous_scale="Cividis_r",

                           range_color=(0,5),

                            mapbox_style="carto-positron",

                           hover_name='county',

                           zoom=2.5, center = {"lat": 37.0902, "lon": -95.7129},

                           labels={'deathrate':'deathrate'}

                          )

fig6.update_layout(margin={"r":0,"t":0,"l":0,"b":0},title_text ='US Covid-19 Deathrate')

fig6.show()
cases_to_population =  pd.read_csv("../input/us-county-covid-casespopulation/casespopulation.csv", dtype={"fips": str})

cases_to_population.at[cases_to_population.loc[cases_to_population['county'] == 'New York City'].index[0],'fips'] = '36061'

cases_to_population.dropna(how = 'any', inplace = True)

cases_to_population.shape

cases_to_population = cases_to_population[['fips','county','case rate']]





cases_to_population['case rate'].describe()
fig7 = px.choropleth_mapbox(cases_to_population, geojson=counties, locations='fips', color='case rate',

                           color_continuous_scale="Tealgrn",

                           range_color=(0,0.11),

                            mapbox_style="carto-positron",

                           hover_name='county',

                           zoom=2.5, center = {"lat": 37.0902, "lon": -95.7129},

                           labels={'case rate':'case rate'}

                          )

fig7.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig7.show()