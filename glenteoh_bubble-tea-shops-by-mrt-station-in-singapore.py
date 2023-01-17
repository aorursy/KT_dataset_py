# Installing the python library for the google maps API
!pip install -U googlemaps 
import googlemaps

#installing the other 'usual' datascience libraries
from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pandas.io.json import json_normalize
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#install folium for map visualisation
!pip install folium
import folium 
!wget -q https://glenteoh.com/datascience/mrt_lrt_coordinates.csv
#credits to Lee Yu Xuan at https://www.kaggle.com/yxlee245/singapore-train-station-coordinates for compiling the coordinates of MRT stations in Singapore
print('Data downloaded!')
mrt_df = pd.read_csv('mrt_lrt_coordinates.csv')
mrt_df.head()
indexes = mrt_df[mrt_df['type'] == "LRT"].index
mrt_df.drop(indexes, inplace=True)
mrt_df = mrt_df.reset_index(drop=True)
mrt_df.tail() #we look at the tail, because the LRT stations are placed at the bottom.
gmaps = googlemaps.Client(key=gmaps_key)
bishan_bubble_tea = gmaps.places_nearby(
            radius=500,
            location=(1.3512096,103.8485599),
            keyword="bubble tea"
        )
bishan_bubble_tea
test_list = []
for store in bishan_bubble_tea['results']:
    test_list.append([
                store['name'],   
                store['rating'],
                store['vicinity'],
                store['geometry']['location']['lat'], 
                store['geometry']['location']['lng']])
df = pd.DataFrame(test_list, columns=["Name", "Rating","Address", "Lat", "Lng"])
df
def getNearbyPlaces(mrt, latitudes, longitudes):
    
    place_list=[]
    for name, lat, lng in zip(mrt, latitudes, longitudes):
        print("Querying " + name)
            
        # create the API request.
        query = gmaps.places_nearby(
            radius=250,
            location=(lat,lng),
            keyword="bubble tea"
        )
        
        for place in query['results']:
            place_list.append([
                        name,
                        place['name'],   
                        place['rating'],
                        place['vicinity'],
                        place['geometry']['location']['lat'], 
                        place['geometry']['location']['lng']])
    places_df = pd.DataFrame(place_list, columns=["MRT_Station", "Name", "Rating","Address", "Lat", "Lng"])
        
    return(places_df)
mrt_bubbletea = getNearbyPlaces(mrt=mrt_df['station_name'],
                                latitudes=mrt_df['lat'],
                                longitudes=mrt_df['lng'])
mrt_bubbletea.tail()
mrt_bubbletea.to_csv(r'mrt_bubble_tea.csv', index = False)
mrt_bubbletea.shape
mrt_bubbletea_counts = mrt_bubbletea.groupby('MRT_Station').count().reset_index()
mrt_bubbletea_counts
mrt_bubbletea_counts.drop(['Rating', 'Address','Lat','Lng'], axis=1, inplace=True)
mrt_bubbletea_counts = mrt_bubbletea_counts.rename(columns={"Name": "Count"})
mrt_bubbletea_counts.sort_values(by=['Count'], ascending=True, inplace=True)
mrt_bubbletea_counts = mrt_bubbletea_counts.set_index('MRT_Station')
mrt_bubbletea_counts.head()
count2 = mrt_bubbletea_counts['Count'].tail(10)
count2
count = mrt_bubbletea_counts['Count']

mpl.style.use('default')
mrt_bubbletea_counts.plot(kind='barh', figsize=(25, 25))
plt.title('Number of Bubble Tea Stores by MRT Station', fontsize=16)

for index, value in enumerate(count): 
    label = format(str(value)) # format int with commas
    plt.annotate(label, xy=(value + 0.1, index - 0.25 ), color='black')
top10 = mrt_bubbletea_counts['Count'].tail(10)

mpl.style.use('default')
mrt_bubbletea_counts.tail(10).plot(kind='barh', figsize=(10, 5))
plt.title('Top 10 MRT Stations by No. of Bubble Tea Shops', fontsize=16)

for index, value in enumerate(top10): 
    label = format(str(value)) # format int with commas
    plt.annotate(label, xy=(value + 0.1, index - 0.15 ), color='black')
bottom10 = mrt_bubbletea_counts['Count'].head(10)

mpl.style.use('default')
mrt_bubbletea_counts.head(10).plot(kind='barh', figsize=(10, 5))
plt.title('Bottom 10 MRT Stations by No. of Bubble Tea Shops', fontsize=16)

mrt_merged = pd.merge(mrt_df, mrt_bubbletea_counts, left_on='station_name', right_on='MRT_Station')
mrt_merged.head()
SIN_lat = '1.3494661'
SIN_lon = '103.8405051'

# create map
map_stores = folium.Map(location=[SIN_lat, SIN_lon],tiles="OpenStreetMap", zoom_start=12)

# set color scheme for the stores
x = np.arange(20)
ys = [i + x + (i*x)**2 for i in range(20)]
colors_array = cm.jet(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, stores in zip(mrt_merged['lat'], mrt_merged['lng'], mrt_merged['station_name'], mrt_merged['Count']):
    label = folium.Popup(str(poi) + " | " + str(stores) + " stores", max_width=200, parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[stores-1],
        fill=True,
        fill_color=rainbow[stores-1],
        fill_opacity=0.9).add_to(map_stores),


map_stores