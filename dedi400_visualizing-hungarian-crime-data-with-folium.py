import requests
import json
import numpy as np
import pandas as pd
import geopandas as gpd #I'm not sure if any functionality of geopandas is needed for this project, but it's cool, an I wanted to check:)
import folium
import folium.plugins
%matplotlib inline
url_base='https://terkep.police.hu/api/odata/v1/'
url_crimes='https://terkep.police.hu/api/odata/v1/Crimes'
url_accidents='https://terkep.police.hu/api/odata/v1/Accidents'
url_infections='https://terkep.police.hu/api/odata/v1/Infections'
resp=requests.get(url_base+"Crime('CRIME_30_ALL_CRIME')") #it is sooo simple to get data with requests library, I love it!
if resp.status_code != 200:
    # This means something went wrong.
    raise requests.HTTPError('Error {}'.format(resp.status_code))

print(resp.headers['content-type'])
resp.encoding = 'UTF-8'
with open('data4.json', 'w') as outfile:
    #I'm saving the data locally just in case, not used in the rest of the notebook
    json.dump(resp.json(), outfile)

crime_json=pd.DataFrame(resp.json())
crime_json.head()
#We actually need data column only
crime=pd.DataFrame(dict(crime_json['data'])).transpose()
crime.head()
crime.shape
crime.dtypes
crime['lat']=crime['lat'].astype(float)
crime['lon']=crime['lon'].astype(float)
crime.dtypes
gdf_crime=gpd.GeoDataFrame(crime, geometry=gpd.points_from_xy(crime['lon'], crime['lat']))
gdf_crime.head()
gdf_crime['Description']=gdf_crime['shortInfo']+'<br>'+gdf_crime['longInfo'].apply(lambda str: (str.split('\\'))[0])
gdf_crime.head()
map=folium.Map(location=[47.16,19.50],zoom_start=7,min_zoom=7)
crime_group=folium.plugins.FastMarkerCluster([]).add_to(map)

for geo,label in zip(gdf_crime['geometry'],gdf_crime['Description']):
    folium.Marker(
        location=[geo.x,geo.y],
        icon=None,
        tooltip=label 
    ).add_to(crime_group)

map.add_child(crime_group)
map
map.save('map.html') #and it can be saved into an html file, keeping all interactivity!