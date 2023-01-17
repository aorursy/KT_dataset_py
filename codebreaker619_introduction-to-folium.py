# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install ipywidgets
!pip install geocoder
!pip install geopy
import folium

m = folium.Map()
m
m = folium.Map(
    location=[26.8177, 75.8617],
    zoom_start = 10,                #Smaller the number - more the map will be zoomed out and vice versa
    #tiles="Stamen Terrain",
    #tiles="Stamen Toner",
    #tiles="Stamen Watercolor",
    #tiles="Mapbox Bright",
    #tiles="Mapbox Control Room",
    #tiles="OpenStreetMap",             #default tile - OpenStreetMap
    #tiles="CartoDB Positron",
    tiles="CartoDB Dark_Matter",
    #tiles="http://{s}.tiles.yourtiles.com/{z}/{x}/{y}.png",     #Custom Tiles

    #min_zoom=20,       
    #max_zoom=90,
    control_scale=True,
    prefer_canvas=True,
    #no_touch=True,
    #disable_3d=True,
    #zoom_control=False,
    #width=500,
    #height=500
    )

m
#Resize the Map

from branca.element import Figure
fig = Figure(width=500, height=500)
fig.add_child(m)
import ipywidgets

#Different Types of Maps - show using ipywidgets

#widget
select_widget = ipywidgets.Select(
    options = ["Open Street Map", "Terrain", "Toner", "Watercolor", "Positron", "Dark Matter"],
    value = "Open Street Map",
    description = "Map Type:",
    disabled = False)

def select(map_type):
  if map_type == "Open Street Map":
    display(folium.Map(location=[26.8177, 75.8617], zoom_start=12, height=700))
  if map_type == "Terrain":
    display(folium.Map(location=[26.8177, 75.8617], tiles="Stamen Terrain",zoom_start=12, height=700))
  if map_type == "Toner":
    display(folium.Map(location=[26.8177, 75.8617], tiles="Stamen Toner",zoom_start=12, height=700))
  if map_type == "Watercolor":
    display(folium.Map(location=[26.8177, 75.8617], tiles="Stamen Watercolor",zoom_start=12, height=700))
  if map_type == "Positron":
    display(folium.Map(location=[26.8177, 75.8617], tiles="CartoDB Positron",zoom_start=12, height=700))
  if map_type == "Dark Matter":
    display(folium.Map(location=[26.8177, 75.8617], tiles="CartoDB Dark_Matter",zoom_start=12, height=700))

ipywidgets.interact(select, map_type=select_widget)
#Show maps using Map Layer Control

map_layer_control = folium.Map(location=[26.8177, 75.8617], zoom_start=2)
folium.raster_layers.TileLayer('Open Street Map').add_to(map_layer_control)
folium.raster_layers.TileLayer('Stamen Terrain').add_to(map_layer_control)
folium.raster_layers.TileLayer('Stamen Toner').add_to(map_layer_control)
folium.raster_layers.TileLayer('Stamen Watercolor').add_to(map_layer_control)
folium.raster_layers.TileLayer('CartoDB Dark_Matter').add_to(map_layer_control)
folium.raster_layers.TileLayer('CartoDB Positron').add_to(map_layer_control)

folium.LayerControl().add_to(map_layer_control)
map_layer_control
#Creating Minimaps
from folium import plugins

m = folium.Map(location=[26.8549, 75.8243], zoom_start=7)
minimap = plugins.MiniMap(toggle_display=True)
m.add_child(minimap)
plugins.ScrollZoomToggler().add_to(m)
plugins.Fullscreen(position="topright").add_to(m)

m
import geocoder
import geopy

#Markers
map_home = folium.Map(
    location=[26.8549, 75.8243],
    zoom_start=14
)
address = geocoder.osm("A-1006, Ashiana Greenwood, Near Shooting Range, Jagatpura, Jaipur")
address_latlng = [address.lat, address.lng]

folium.Marker([26.8549, 75.8243], 
              popup='<i>Ashiana Greenwood</i>',
              tooltip = "Click Me!"
              ).add_to(map_home)

map_home
m = folium.Map(
    location=[26.8549, 75.8243],
    zoom_start=12,
    tiles='Stamen Terrain'
)

tooltip = 'Click me!'

folium.Marker([26.8549, 75.8243], 
              popup='<i>Gaurav Tower</i>', 
              tooltip=tooltip,
              icon=folium.Icon(icon='cloud', color="red", icon_color="blue")
              ).add_to(m)
#folium.Marker([26.8177, -75.8617], popup='<b>Malviya Nagar</b>', tooltip=tooltip).add_to(m)

m
#font-awesome Custom Icon
#website - https://fontawesome.com/icons?d=gallery

mp = folium.Map(
    location=[26.8549, 75.8243],
    zoom_start=12,
    tiles='Stamen Terrain'
)

tooltip = 'Click me!'

folium.Marker([26.8549, 75.8243], 
              popup='Jaipur', 
              tooltip=tooltip,
              icon=folium.Icon(icon='bolt', color="green", icon_color="white",prefix="fa")
              ).add_to(mp)

mp
#Using Glyphicon Bootstrap icons
#website - https://getbootstrap.com/docs/3.3/components/

mp_cm_glyphicon = folium.Map(location=[26.8549, 75.8243],
    zoom_start=12,
    tiles='Stamen Terrain')

folium.Marker(location=[26.8549, 75.8243],
              popup="city",
              icon=folium.Icon(icon="glyphicon-plane", prefix="glyphicon")).add_to(mp_cm_glyphicon)

mp_cm_glyphicon
mp_circle = folium.Map(
    location=[26.8549, 75.8243],
    zoom_start=12,
    tiles='Stamen Terrain'
)

#Circle gets bigger, smaller on zoomin, zoomout
folium.Circle(
    radius=1000,          #radius in meters
    location=[26.8586, 75.8243],
    popup='The Mall',
    color='crimson',
    zoom_start=40,
    fill=False
).add_to(mp_circle)

mp_circle
mp_circle = folium.Map(
    location=[26.8549, 75.8243],
    zoom_start=12,
    tiles='CartoDB Dark_Matter'
)

#Circle Stays the same size
folium.CircleMarker(
    location=[26.8999, 75.8243],
    radius=50,              #radius in pixels
    popup='Raja Park',
    color='blue',
    fill=True,
    fill_color='blue',
    fill_opacity = 0.3
).add_to(mp_circle)

mp_circle
#Enble Lat Long Popovers - find location interactively browsing the map
m = folium.Map(
    location=[26.8999, 75.8243],
    zoom_start=13
)

m.add_child(folium.LatLngPopup())

m
#On the Fly Placement of Markers
m = folium.Map(
    location=[26.8999, 75.8243],
    tiles='Stamen Terrain',
    zoom_start=13
)

folium.Marker(
    [26.8999, 75.8243],
    popup='Raja Park'
).add_to(m)

m.add_child(folium.ClickForMarker(popup='Waypoint'))

m
#Creating Route

m_route = folium.Map(
    location=[26.8999, 75.8243],
    tiles = "CartoDB Dark_Matter",
    zoom_start=13
)

route_lats_long = [[26.9058, 75.7873],
                   [26.8892, 75.8039],
                   [26.8763, 75.8122],
                   [26.8676, 75.7921]]

#Add route to map
folium.PolyLine(route_lats_long).add_to(m_route)

m_route
#Creating Route using antpath - Animated path

mp_ant_route = folium.Map(
    location=[26.8999, 75.8243],
    tiles = "CartoDB Dark_Matter",
    zoom_start=13
)

route_lats_long = [[26.8905, 75.7602],
                   [26.8679, 75.8427],
                   [26.9172, 75.7365],
                   [26.8676, 75.7921]]

plugins.AntPath(route_lats_long).add_to(mp_ant_route)

mp_ant_route
!pip install vega_datasets
#Airports dataframe using vega_datasets
from vega_datasets import data as vds
airports = vds.airports()
airports = airports[:25]
airports.head()

#Locating Airports in United States
map_airports = folium.Map(location=[38,-98],zoom_start=4)

#Plot airport locations - method 1 - using loop
for (index,row) in airports.iterrows():
  folium.Marker(location = [row.loc['latitude'], row.loc['longitude']], 
              popup=row.loc['name']+' '+row.loc['city']+' '+row.loc['state'], 
              tooltip="Click Me!",
              icon=folium.Icon(icon="glyphicon-plane", prefix="glyphicon")
              ).add_to(map_airports)

#Plot airport locations - method 2 - without using loop
#map_airports_2 = folium.Map(location=[38,-98],zoom_start=4)

#airports.apply(lambda row: folium.Marker(location = [row.loc['latitude'], row.loc['longitude']],
                                         #popup=row.loc['name'],
                                         #icon=folium.Icon(icon="glyphicon-plane", prefix="glyphicon")).add_to(map_airports_2), axis=1)

map_airports
#If the dataset is a dictionary

markers_dict = {"Los Angeles":[34.041008, -118.246653],
                "Las Vegas":[36.169726, -115.143996],
                "Denver":[39.739448, -104.992450],
                "Chicago":[41.878765, -87.643267],
                "Manhattan":[40.782949, -73.969559]}

map_cities = folium.Map(location=[41, -99], zoom_start=4)

for i in markers_dict.items():
  folium.Marker(location=i[1], popup=i[0]).add_to(map_cities)
  print(i)

map_cities
#Multiple Custom Markers
import pandas as pd

cm_df = pd.DataFrame({'city' : ['Los Angeles', 'Las Vegas', 'Denver', 'Chicago', 'Manhattan'],
                      'latitude' : [34.041008, 36.169726, 39.739448, 41.878765, 40.782949],
                      'longitude' : [-118.246653, -115.143996, -104.992450, -87.643267, -73.969559],
                      'icon' : ['bicycle','car','bus','truck','motorcycle']})
cm_df
mp_multiple_mark = folium.Map(location=[38,-98], zoom_start=4)

for i in cm_df.itertuples():
  folium.Marker(location=[i.latitude, i.longitude],
                popup=i.city,
                icon = folium.Icon(icon=i.icon, prefix='fa')).add_to(mp_multiple_mark)

mp_multiple_mark
for i in cm_df.itertuples():
  print(i)
  print(i.city)
cm_df = pd.DataFrame({'city' : ['Los Angeles', 'Las Vegas', 'Denver', 'Chicago', 'Manhattan'],
                      'latitude' : [34.041008, 36.169726, 39.739448, 41.878765, 40.782949],
                      'longitude' : [-118.246653, -115.143996, -104.992450, -87.643267, -73.969559],
                      'icon_num' : [1,2,3,4,5]})

map_enum_icons = folium.Map([38, -98], zoom_start=4)

for i in cm_df.itertuples():
  folium.Marker(location=[i.latitude, i.longitude],
                popup=i.city,
                icon = plugins.BeautifyIcon(number=i.icon_num,
                                            border_color='blue',
                                            border_width=1,
                                            text_color='red',
                                            inner_icon_style='margin-top:0px;',
                                            #icon="graphicon-plane",
                                            #border_width=3,
                                            #border_color='#000',
                                            #background_color="#FFF",
                                            #inner_color_style="",
                                            spin=True,
                                            )).add_to(map_enum_icons)

map_enum_icons
cm_df
#Overlay GeoJSON Layers 
import json
map_geoJson = folium.Map(location=[26.8999, 75.8243], zoom_start=11)
folium.GeoJson(json.load(open('/kaggle/input/jaipur-map/map.geojson')), name="geojson Jaipur").add_to(map_geoJson)
folium.LayerControl().add_to(map_geoJson)
map_geoJson
#Creating HeatMap

#location data for large cities in India
Mumbai = geocoder.osm("Mumbai, Maharashtra, India")
Kolkata = geocoder.osm("Kolkata, West Bengal, India")
New_Delhi = geocoder.osm("New Delhi, Delhi, India")
Chennai = geocoder.osm("Chennai, Tamil Nadu, India")
Bangalore = geocoder.osm("Bangalore, Karnataka, India")
Jaipur = geocoder.osm("Jaipur, Rajasthan, India")
Kerala = geocoder.osm("Kerala, India")
Hyderabad = geocoder.osm("Hyderabad, Telangana, India")
Ahmedabad = geocoder.osm("Ahmedabad, Gujarat, India")

#Create Latitude, Longitude, Intensity of Population
Mumbai_latlang = [Mumbai.lat, Mumbai.lng, 18400000/10000]
Kolkata_latlang = [Kolkata.lat, Kolkata.lng, 4500000/10000]
New_Delhi_latlang = [New_Delhi.lat, New_Delhi.lng, 21800000/10000]
Chennai_latlang = [Chennai.lat, Chennai.lng, 7090000/10000]
Bangalore_latlang = [Bangalore.lat, Bangalore.lng, 8430000/10000]
Jaipur_latlang = [Jaipur.lat, Jaipur.lng, 3070000/10000]
Kerala_latlang = [Kerala.lat, Kerala.lng, 34800000/10000]
Hyderabad_latlang = [Hyderabad.lat, Hyderabad.lng, 6810000/10000]
Ahmedabad_latlang = [Ahmedabad.lat, Ahmedabad.lng, 5570000/10000]

#Create list of Large Cities
large_cities = [Mumbai_latlang, Kolkata_latlang, New_Delhi_latlang, Chennai_latlang, Bangalore_latlang, Jaipur_latlang, Kerala_latlang, Hyderabad_latlang,
                Ahmedabad_latlang]

map_heatmap = folium.Map([20.5937, 78.9629], tiles="CartoDB Dark_Matter", zoom_start=5)

plugins.HeatMap(large_cities, 
                name="Population Density", 
                #radius=50,
                #min_opacity = 1,
                #max_zoom=18,
                #blur=20,
                #overlay=False,
                #control=False,
                #show=False
                ).add_to(map_heatmap)

map_heatmap
import numpy as np
import pandas as pd

heatmap_time_data = (np.random.random((100,20,2)) + np.array([[20.5937, 78.9629]])).tolist()
heatmap_time_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range('20160101', periods=len(heatmap_time_data))]
map_heatmap_time = folium.Map([20.5937, 78.9629], tiles="CartoDB Dark_Matter",zoom_start=6)
heatmap_time_plugin = plugins.HeatMapWithTime(heatmap_time_data, index=heatmap_time_dates)
heatmap_time_plugin.add_to(map_heatmap_time)

map_heatmap_time
#Measure Control

map_Measure = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB Dark_Matter")

Measure_Control = plugins.MeasureControl(position='topright', 
                                         #active_color="red",
                                         #completed_color="red",
                                         primary_length_unit='miles', 
                                         secondary_length_unit='miles', 
                                         primary_area_unit='sqmeters', 
                                         secondary_area_unit='acres')

map_Measure.add_child(Measure_Control)
map_Measure
#Dual Map

map_dual = plugins.DualMap(location=[20.5937, 78.9629], zoom_start=4, layout="vertical")

folium.TileLayer("Stamen Terrain").add_to(map_dual)
folium.TileLayer("CartoDB Dark_Matter").add_to(map_dual)

folium.LayerControl().add_to(map_dual)

map_dual
#Draw

map_draw = folium.Map(location=[20.5937, 78.9629], zoom_start=4, tiles="CartoDB Dark_Matter")
draw = plugins.Draw(export=True).add_to(map_draw)
map_draw
#Overlay Image

map_image_overlay = folium.Map([2, 22], zoom_start=2, tiles="CartoDB Dark_Matter")

img_overlay = folium.raster_layers.ImageOverlay(name="Flags of Africa",
                                                image="/kaggle/input/flags-of-africa-and-asia/flags_of_Africa.png",
                                                bounds=[[-38,-28],[40, 60]],
                                                opacity=0.3,
                                                zindex=1)

img_overlay.add_to(map_image_overlay)
folium.LayerControl().add_to(map_image_overlay)

map_image_overlay
#Charts in PopUP

seattle_weather = vds.seattle_weather()
seattle_weather.head()
import altair

sw_means = pd.DataFrame({"seattle_x": ['precipitation',	'temp_max',	'temp_min',	'wind'],
                        "seattle_y": [seattle_weather.precipitation.mean(),
                                     seattle_weather.temp_max.mean(),
                                     seattle_weather.temp_min.mean(),
                                     seattle_weather.wind.mean()]})

chart = altair.Chart(sw_means, width=300).mark_bar().encode(x="seattle_x",y="seattle_y").properties(title="Seattle Weather Averages")

chart
#Embed Chart in marker corresponding to location

vega = folium.features.VegaLite(chart, width="100%", height="100%")
map_chart = folium.Map(location=[47.606322, -122.332575])
marker = folium.features.Marker([47.60, -122.33])
popup = folium.Popup()
vega.add_to(popup)
marker.add_to(map_chart)
popup.add_to(marker)

map_chart
address_box = ipywidgets.Text(valur='', placeholder='type here', description='address')
def plot_locations(address):
  location=geocoder.osm(address)
  latlng = [location.lat, location.lng]
  plot_locations_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
  folium.Marker(latlng, popup=str(address), tooltip="Click").add_to(plot_locations_map)
  display(plot_locations_map)

ipywidgets.interact_manual(plot_locations, address=address_box)
import geopy.distance

route_start_widget = ipywidgets.Text(valur='', placeholder='type here', description='start:')
route_stop_widget = ipywidgets.Text(valur='', placeholder='type here', description='stop:')

def get_distance(start_address, stop_address):
  start_location = geocoder.osm(start_address)
  stop_location = geocoder.osm(stop_address)

  start_latlng = [start_location.lat, start_location.lng]
  stop_latlng = [stop_location.lat, stop_location.lng]

  distance = geopy.distance.distance(start_latlng, stop_latlng).miles
  print(f"Distance: {distance:.2f} miles")

  distance_path = [(start_latlng), (stop_latlng)]
  map_distance  = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
  plugins.AntPath(distance_path).add_to(map_distance)
  display(map_distance)

ipywidgets.interact_manual(get_distance, start_address=route_start_widget, stop_address=route_stop_widget)
#FeatureGroupSubGroup

map_with_subgroups = folium.Map(location=[39.77, -86.15], zoom_start=7, tiles="CartoDB Dark_Matter")

# all subgroups
all_subgroups = folium.FeatureGroup(name='all sales regions')
map_with_subgroups.add_child(all_subgroups)

# subgroup 1
sales_region1 = plugins.FeatureGroupSubGroup(all_subgroups, 'sales region 1')
map_with_subgroups.add_child(sales_region1)

# subgroup 2
sales_region2 = plugins.FeatureGroupSubGroup(all_subgroups, 'sales region 2')
map_with_subgroups.add_child(sales_region2)

# subgroup 3
sales_region3 = plugins.FeatureGroupSubGroup(all_subgroups, 'sales region 3')
map_with_subgroups.add_child(sales_region3)

# pull in geojson layers and add to map
folium.GeoJson(json.load(open('/kaggle/input/region-sales/sales_region1.geojson')),name="Sales Region 1").add_to(sales_region1)
folium.GeoJson(json.load(open('/kaggle/input/region-sales/sales_region2.geojson')), name="Sales Region 2").add_to(sales_region2)
folium.GeoJson(json.load(open('/kaggle/input/region-sales/sales_region3.geojson')), name="Sales Region 3").add_to(sales_region3)

# add layer control to map (allows layers to be turned on or off)
folium.LayerControl(collapsed=False).add_to(map_with_subgroups)

# display map
map_with_subgroups