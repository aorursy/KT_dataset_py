import math

import pandas as pd

import geopandas as gpd



import folium

from folium import Figure, Map, Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster
# Load the data

crimes = pd.read_csv("../input/geospatial-learn-course-data/crimes-in-boston/crimes-in-boston/crime.csv", encoding='latin-1')



# Drop rows with missing locations

crimes.dropna(subset=['Lat', 'Long', 'DISTRICT'], inplace=True)



# Focus on major crimes in 2018

crimes = crimes[crimes.OFFENSE_CODE_GROUP.isin([

    'Larceny', 'Auto Theft', 'Robbery', 'Larceny From Motor Vehicle', 'Residential Burglary',

    'Simple Assault', 'Harassment', 'Ballistics', 'Aggravated Assault', 'Other Burglary', 

    'Arson', 'Commercial Burglary', 'HOME INVASION', 'Homicide', 'Criminal Harassment', 

    'Manslaughter'])]

crimes = crimes[crimes.YEAR>=2018]



day_rob = crimes[((crimes.OFFENSE_CODE_GROUP == 'Robbery') & \

                            (crimes.HOUR.isin(range(9,18))))]



# Print the first five rows of the table

day_rob.head()
# folium.Map?
# folium.Figure?
# f.add_child?
# openstreet map

f = Figure(width=400, height=250, title='Open Street Map') # width and height of openstreet map

m1 = Map(location=[9.9312, 76.2673],    # coordinates gives the center of the map

                tiles='openstreetmap',  # map style

                zoom_start=12)          # starting zoom value

f.add_child(m1)

f
# cartodbpositron map

f = Figure(width=400, height=250)

m1 = Map(location=[9.9312, 76.2673], 

                tiles='cartodbpositron', 

                zoom_start=12)

f.add_child(m1)

f
day_rob.head()
m2 = Map(location=[42.32,-71.0589], 

                 tiles='cartodbpositron', 

                 zoom_start=13)



for idx, row in day_rob.iterrows(): # iterate over pandas rows 

    

    if row['OFFENSE_DESCRIPTION']=='ROBBERY - STREET':

        Marker([row['Lat'], row['Long']],

               popup = row['OFFENSE_DESCRIPTION'], # appears on click

               tooltip=row['STREET'],              # appears on hover

               icon=folium.Icon(color='blue', icon='cloud')).add_to(m2)

        

    elif row['OFFENSE_DESCRIPTION']=='ROBBERY - COMMERCIAL':

        Marker([row['Lat'], row['Long']],

               popup = row['OFFENSE_DESCRIPTION'], # appears on click

               tooltip=row['STREET'],              # appears on hover

               icon=folium.Icon(color='green', icon='home', prefix='fa')).add_to(m2)

        

    else:

        Marker([row['Lat'], row['Long']], 

               popup = row['OFFENSE_DESCRIPTION'],

               tooltip=row['STREET'],

               icon=folium.Icon(color='red', icon='info-sign')).add_to(m2)

    

m2
# Circle?
m2 = Map(location=[42.32,-71.0589], 

                 tiles='cartodbpositron', 

                 zoom_start=13)



for idx, row in day_rob.iterrows(): # iterate over pandas rows 



    Circle([row['Lat'], row['Long']],

           popup = row['OFFENSE_DESCRIPTION'], # appears on click

           tooltip=row['STREET'],              # appears on hover

           radius = 100,    # radius in meters

           color='#3186cc', # color of the circumference circle

           fill=True, 

           fill_color='#3186cc').add_to(m2)

    

m2
m3 = Map(location=[42.32, -71.0789], tiles='cartodbpositron', zoom_start=13)

mc = MarkerCluster()

for idx, row in day_rob.iterrows():

    if not math.isnan(row['Long']) and not math.isnan(row['Lat']):

        mc.add_child(Marker([row['Lat'], row['Long']]))

m3.add_child(mc)

m3
m4 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)



def color_producer(val):

    if val <= 12:

        return 'forestgreen'

    else:

        return 'darkred'



for i in range(0,len(day_rob)):

    Circle(location=[day_rob.iloc[i]['Lat'], day_rob.iloc[i]['Long']],

           radius=20, 

           color=color_producer(day_rob.iloc[i]['HOUR'])).add_to(m4)



m4
m5 = Map(location=[42.32, -71.0589], title='cartodbpositron', zoom_start=12)

HeatMap(data=crimes[['Lat', 'Long']], radius=10).add_to(m5)

m5
# GeoDataFrame with geographical boundaries of Boston police districts

districts_full = gpd.read_file('../input/geospatial-learn-course-data/Police_Districts/Police_Districts/Police_Districts.shp')

districts = districts_full[["DISTRICT", "geometry"]].set_index("DISTRICT")

print(districts.head())



plot_dict = crimes.DISTRICT.value_counts()

print(plot_dict.head())



m6 = folium.Map(location=[42.32,-71.0589], 

                tiles='cartodbpositron', 

                zoom_start=12)



# Add a choropleth map to the base map

Choropleth(geo_data=districts.__geo_interface__, 

           data=plot_dict, 

           key_on="feature.id", 

           fill_color='YlGnBu', 

           legend_name='Major criminal incidents (Jan-Aug 2018)').add_to(m6)



# Display the map

m6
m = folium.Map(location=[45.523, -122.675], width=750, height=500)

m.fit_bounds([[52.193636, -2.221575], [52.636878, -1.139759]])

m
folium.Map(location=[45.523, -122.675], tiles='Mapbox Control Room')