import pandas as pd

import geopandas as gpd

import math
import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster
m_1 = folium.Map(location=[42.32,-71.0589], tiles='openstreetmap', zoom_start=10)

m_1
crimes = pd.read_csv("../input/crimes-in-boston/crime.csv", encoding='latin-1')



# Drop rows with missing locations

crimes.dropna(subset=['Lat', 'Long', 'DISTRICT'], inplace=True)



# Focus on major crimes in 2018

crimes = crimes[crimes.OFFENSE_CODE_GROUP.isin([

    'Larceny', 'Auto Theft', 'Robbery', 'Larceny From Motor Vehicle', 'Residential Burglary',

    'Simple Assault', 'Harassment', 'Ballistics', 'Aggravated Assault', 'Other Burglary', 

    'Arson', 'Commercial Burglary', 'HOME INVASION', 'Homicide', 'Criminal Harassment', 

    'Manslaughter'])]

crimes = crimes[crimes.YEAR>=2018]



# Print the first five rows of the table

crimes.head()
daytime_robberies = crimes[((crimes.OFFENSE_CODE_GROUP == 'Robbery') & \

                            (crimes.HOUR.isin(range(7,23))))]
m_2 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)



for idx, row in daytime_robberies.iterrows():

    Marker([row['Lat'], row['Long']]).add_to(m_2)

m_2
m_3 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)



mc = MarkerCluster()

for idx, row in daytime_robberies.iterrows():

    if not math.isnan(row['Long']) and not math.isnan(row['Lat']):

        mc.add_child(Marker([row['Lat'], row['Long']]))

m_3.add_child(mc)



# Display the map

m_3

m_4 = folium.Map(location=[42.32,-71.0589], tiles="cartodbpositron",zoom_start=15)

def color_val(val):

    return 'forestgreen' if (val<=12) else "darkred"

    

for i in range(0,len(daytime_robberies)):

    Circle(

        location=[daytime_robberies.iloc[i]['Lat'], daytime_robberies.iloc[i]['Long']],

        radius=40,

        color=color_val(daytime_robberies.iloc[i]['HOUR'])).add_to(m_4)

m_4
m_5 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=12)

HeatMap(data=crimes[['Lat', 'Long']], radius=10).add_to(m_5)

m_5
# GeoDataFrame with geographical boundaries of Boston police districts

districts_full = gpd.read_file('../input/boston-police-districts/Police_Districts.shp')

districts = districts_full[["DISTRICT", "geometry"]].set_index("DISTRICT")

districts.head()
plot_dict = crimes.DISTRICT.value_counts()

plot_dict.head()
m_6 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=12)



Choropleth(geo_data=districts.__geo_interface__, data=plot_dict, key_on="feature.id", fill_color='YlGnBu', 

           legend_name='Major criminal incidents (Jan-Aug 2018)').add_to(m_6)

m_6