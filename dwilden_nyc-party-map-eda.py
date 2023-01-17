from subprocess import check_output

print(check_output(["ls", "../input/nyc-neighborhoods"]).decode("utf8"))
import numpy as np

import pandas as pd

import folium

from folium import features

from folium.plugins import HeatMap

from folium.plugins import MarkerCluster



import seaborn as sns

import time

import datetime

import matplotlib.pyplot as plt

%matplotlib inline
bars = pd.read_csv('../input/partynyc/bar_locations.csv')

bars.head()
mlat = bars['Latitude'].median()

mlon = bars['Longitude'].median()

print(mlat, mlon)
'''

import geopandas as gpd

import json

import rtree

from shapely.geometry import Point



#Takes a geoJson and converts it into a GeoDataFrame

district_geo =r'hoods.json'

data2 = json.load(open(district_geo))

data3 = gpd.GeoDataFrame.from_features(data2['features'])

#Turns the bar data into a GeoDataFrame

bars['geometry']= [Point(xy) for xy in zip(bars.Longitude, bars.Latitude)]

crs = {'init': 'epsg:4326'}

gdf = gpd.GeoDataFrame(bars, crs=crs, geometry=bars['geometry'])

#Spatially joins the two GeoDataFrames determining if a bar is in a neighborhood

bars_with_hood = gpd.sjoin(gdf, data3, how="right", op='intersects')

#Sums each neighborhoods calls now that the bars have a neighborhood

calls_by_district = bars_with_hood.dissolve(by="neighborhood", aggfunc='sum')

calls_by_district.fillna(0, inplace=True)

#Creates a representative point in each neighborhood

calls_by_district['coords'] = calls_by_district['geometry'].apply(lambda x: x.representative_point().coords[:])

calls_by_district['coords'] = [coords[0] for coords in calls_by_district['coords']]

'''
#Neighborhood json

district_geo =r'../input/nyc-neighborhoods/hoods.json'



#pulls in the result from aggregation, indexes to neighborhood

calls_by_district = pd.read_csv('../input/nyc-neighborhoods/calls_by_district.csv') 

calls_by_district = calls_by_district.set_index('neighborhood')

calls =calls_by_district.iloc[:,5]



#build the map

map_test = folium.Map(location=[mlat, mlon], zoom_start=12)

map_test.choropleth(district_geo, data=calls, key_on='properties.neighborhood',

                    fill_color='YlGn',fill_opacity=0.7,line_opacity=0.2)
#which neighborhood had the most calls?

trial = calls_by_district.nlargest(29, 'num_calls')

trial.head()
map_test
# X marks the spot.. well I gues O does here

for x in range (1,4):

    folium.CircleMarker([40.711662220917816, -73.9505630158169],

                        radius=x*10,

                        popup='Williamsburg',

                        color='Red',

                        fill_color='#3186cc',

                       ).add_to(map_test)

folium.Marker([40.711662220917816, -73.9505630158169],

                    popup='Williamsburg',

                   ).add_to(map_test)

for x in range (1,3):

    folium.CircleMarker([40.72629678410688, -73.98175826912397],

                        radius=x*10,

                        popup='East Village',

                        color='Red',

                        fill_color='#3186cc',

                       ).add_to(map_test)

folium.Marker([40.72629678410688, -73.98175826912397],

                    popup='East Village',

                   ).add_to(map_test)

map_test
bars_with_hood = pd.read_csv('../input/nyc-neighborhoods/bars_with_hood.csv')

bars_with_hood.head()
#Stats for the worst neighborhood

wb_bars=bars_with_hood.loc[bars_with_hood['neighborhood'] == 'Williamsburg']

wb_bars['num_calls'].describe()
import re



# Let's Draw a another map and isolate the really loud bars

check = folium.Map(location=[40.711662220917816, -73.9505630158169], zoom_start=14)

#redrawing the neighborhood boundary

thing = calls_by_district.loc['Williamsburg']['geometry']

#Because this data was pulled in via CSV it lost the polygon datatype

#Used regex to pull the coordinates out of a string to make a new list

pat = re.compile(r'''(-*\d+\.\d+ -*\d+\.\d+);*''')

matches = pat.findall(thing)

if matches:

    lst = [tuple(map(float, m.split())) for m in matches]

thing= lst



#With new list in hand fed the coordinates to folium to draw the boundary

points = []

for x in range(0, len(thing)):

    points.append([thing[x][1],thing[x][0]])

folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(check)

check
mc = MarkerCluster()

for ind, row in wb_bars.iterrows():

    num = row['num_calls']

    if num > 71:

        num = str(num)

        folium.Marker(location=[row['Latitude'], row['Longitude']], popup=num,

                                   icon=folium.Icon(color='red',icon='info-sign')).add_to(check)

    elif num > 33 and num <=71:

        num = str(num)

        mc.add_child(folium.Marker(location=[row['Latitude'], row['Longitude']], popup=num,

                                   icon=folium.Icon(color='orange',icon='info-sign')))

check.add_child(mc)



folium.CircleMarker(location=[40.7178, -73.9577], line_color='Blue', fill_color='Blue', radius=20).add_to(check)

check