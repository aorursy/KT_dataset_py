import pandas as pd

import numpy as np

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster
# Importing csv file with cases report by city

cities_cases = '../input/corona-virus-brazil/brazil_covid19_cities.csv'

df_cities_cases = pd.read_csv(cities_cases, encoding='utf8', parse_dates=True)

#df_cities_cases.head() #df overview

# ---------------------

df_cities_cases_lastdate = df_cities_cases.loc[(df_cities_cases['date'] == df_cities_cases['date'].max())]

#df_cities_cases_lastdate.head() #df overview

# ---------------------

df_last_report_cases = df_cities_cases_lastdate.pivot_table(index=['code'], values=['cases','deaths']).reset_index()

#df_last_report_cases = df_cities_cases_lastdate.pivot_table(index=['code','name','state'], values=['cases']).reset_index()

#df_last_report_cases.head() #df overview

# ---------------------

print('Cases on {0}: {1}'.format(df_cities_cases['date'].max(),df_cities_cases_lastdate['cases'].sum()))

print('Deaths on {0}: {1}'.format(df_cities_cases['date'].max(),df_cities_cases_lastdate['deaths'].sum()))

print('Mortality rate: {0}%'.format(round( (df_cities_cases_lastdate['deaths'].sum()/df_cities_cases_lastdate['cases'].sum()) , ndigits=3)*100))

print('- - - - - - - - - -')

# ---------------------

# Importing csv file with cities geographic datas (latitude and longitude)

coord = '../input/corona-virus-brazil/brazil_cities_coordinates.csv'

df_coord = pd.read_csv(coord, encoding='utf8')

df_coord['city_code'] = df_coord['city_code']//10

df_coord = pd.DataFrame(df_coord, columns=['city_code','lat','long'])

#df_coord.head() #df overview

# ---------------------

# combining file with last cases reports and cities latitude and longitude

df_map_cases = pd.merge(df_last_report_cases ,df_coord ,left_on='code' ,right_on='city_code')

df_map_cases = df_map_cases.drop(columns='city_code')

df_map_cases #df overview

mp01 = folium.Map(location=[-15,-55], tiles='OpenStreetMap', zoom_start=4)

mp01.add_child(HeatMap(zip(df_map_cases['lat'],df_map_cases['long'],df_map_cases['cases']), radius = 9,min_opacity = 0.30 ,

        gradient={.25: 'black', .5: 'yellow', .7: 'orange', .85: 'red', 1: 'darkred'}))

# Using 'cases' to weight heatmap

mp01
mp02 = folium.Map(location=[-15,-55], tiles='OpenStreetMap', zoom_start=4)

mp02.add_child(HeatMap(zip(df_map_cases['lat'],df_map_cases['long'],df_map_cases['deaths']), radius = 8,min_opacity = 0.50 ,

        gradient={.25: 'black', .5: 'yellow', .7: 'orange', .85: 'red', 1: 'darkred'}))

# Using 'cases' to weight heatmap

mp02