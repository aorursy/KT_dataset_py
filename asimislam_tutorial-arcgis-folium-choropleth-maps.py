import pandas as pd

import warnings

warnings.filterwarnings('ignore')



#  Kaggle directories

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#import geocoder   #  geocoder with ArcGIS  <- NOT AVAILABLE ON KAGGLE

import folium      #  folium libraries

from   folium.plugins import MarkerCluster
#  geocoder.arcgis('dallas, texas').latlng    # get lat & lng coord
'''

def arc_latlng(location):

    g = geocoder.arcgis('{}'.format(location))

    lat_lng_coords = g.latlng

    print(location,lat_lng_coords)

    return lat_lng_coords



arc_latlng('dallas, texas')     #  test arc_latlng

'''
'''

#  location list

#  10001 is the zip code of Manhattan, New York, US

#  M9B   is a postal code in Toronto, Canada

#  Everest is Mt. Everest in Nepal

location = ['10001','Tokyo','Sydney','Beijing','Karachi','Dehli', 'Everest','M9B','Eiffel Tower','Sao Paulo','Moscow']





#  call get_latlng function

loc_latlng = [arc_latlng(location) for location in location]





#  create dataframe for the results

df = pd.DataFrame(data = loc_latlng, columns = {'Latitude','Longitude'})

df.columns = ['Latitude','Longitude']  #  correct column order

df['Location'] = location              #  add location names

'''
'''

invalid_loc = ['london','berlin','0902iuey7','999paris']  # 3 & 4 are invalid

invalid_latlng = [arc_latlng(invalid_loc) for invalid_loc in invalid_loc]

'''
Location  = ['10001', 'Tokyo', 'Sydney', 'Beijing', 'Karachi', 'Dehli', 'Everest', 'M9B', 'Eiffel Tower', 'Sao Paulo', 'Moscow']

Latitude  = [40.74876000000006, 35.68945633200008, -33.869599999999934, 39.90750000000003, 24.90560000000005, 28.653810000000078, 27.987910000000056, 43.64969222700006, 48.85859991892235, -23.562869999999975, 55.75696000000005]

Longitude = [-73.99331999999998, 139.69171608500005, 151.2069100000001, 116.39723000000004, 67.08220000000006, 77.22897000000006, 86.92529000000007, -79.55394499999994, 2.293980070546176, -46.654679999999985, 37.61502000000007]



df = pd.DataFrame(columns = {'Location','Latitude','Longitude'})

df.columns = ['Location','Latitude','Longitude']

df['Location']  = Location

df['Latitude']  = Latitude

df['Longitude'] = Longitude



df
#  center map on mean of Latitude/Longitude

map_world = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()], tiles = 'stamenterrain', zoom_start = 2)



#  add Locations to map

for lat, lng, label in zip(df.Latitude, df.Longitude, df.Location):

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        fill=True,

        color='Blue',

        fill_color='Yellow',

        fill_opacity=0.6

        ).add_to(map_world)



#  display interactive map

map_world



#  save map to local machine, open in any browser

#  map_world.save("C:\\ ... <path> ... \map_world_NYC.html")
LocationNY  = ['Empire State Building', 'Central Park', 'Wall Street', 'Brooklyn Bridge', 'Statue of Liberty', 'Rockefeller Center', 'Guggenheim Museum', 'Metlife Building', 'Times Square', 'United Nations Headquarters', 'Carnegie Hall']

LatitudeNY  = [40.74837000000008, 40.76746000000003, 40.705790000000036, 40.70765000000006, 40.68969000000004, 40.758290000000045, 40.78300000000007, 40.75407000000007, 40.75648000000007, 40.74967000000004, 40.76494993060773]

LongitudeNY = [-73.98463999999996, -73.97070999999994, -74.00987999999995, -73.99890999999997, -74.04358999999994, -73.97750999999994, -73.95899999999995, -73.97637999999995, -73.98617999999993, -73.96916999999996, -73.9804299522477]



dfNY = pd.DataFrame(columns = {'Location','Latitude','Longitude'})

dfNY.columns = ['Location','Latitude','Longitude']

dfNY['Location']  = LocationNY

dfNY['Latitude']  = LatitudeNY

dfNY['Longitude'] = LongitudeNY



dfNY
map_world_NYC = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()],

                       tiles = 'openstreetmap', 

                       zoom_start = 1)



#  CIRCLE MARKERS

#------------------------------

for lat, lng, label in zip(df.Latitude, df.Longitude, df.Location):

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        fill=True,

        color='black',

        fill_color='red',

        fill_opacity=0.6

        ).add_to(map_world_NYC)

#------------------------------



    

#  MARKERS CLUSTERS

#------------------------------

marker_cluster = MarkerCluster().add_to(map_world_NYC)

for lat, lng, label in zip(dfNY.Latitude, dfNY.Longitude, dfNY.Location):

    folium.Marker(location=[lat,lng],

            popup = label,

            icon = folium.Icon(color='green')

    ).add_to(marker_cluster)



map_world_NYC.add_child(marker_cluster)

#------------------------------



#  display map

map_world_NYC         
dfs = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')  # suicide rates dataset



dfs = dfs[dfs['year'] == 2013]

dfs = dfs[['country','year','suicides/100k pop']].groupby('country').sum()

dfs.reset_index(inplace=True)



#  update names to match names in geoJSON file

dfs.replace({

        'United States':'United States of America',

        'Republic of Korea':'South Korea',

        'Russian Federation':'Russia'},

        inplace=True)



dfs.head()
world_geo = os.path.join('../input/worldcountries', 'world-countries.json')

world_geo
world_choropelth = folium.Map(location=[0, 0], tiles='Mapbox Bright',zoom_start=1)



world_choropelth.choropleth(

    geo_data=world_geo,

    data=dfs,

    columns=['country','suicides/100k pop'],

    key_on='feature.properties.name',

    fill_color='YlOrRd',

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Suicide rates per 100k Population - 2013')



folium.LayerControl().add_to(world_choropelth)

# display map

world_choropelth