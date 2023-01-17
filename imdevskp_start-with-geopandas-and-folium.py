# data storing and anaysis

import pandas as pd

import geopandas as gpd



# visualization

import matplotlib.pyplot as plt

import folium

import plotly.express as px
# import shape file

dist = gpd.read_file('../input/output.shp')

dist.head()
# plot the geopands dataframe

dist.plot() # geopandas looks for the geometry column to plot individual piece in the map
fig, ax = plt.subplots(figsize=(12, 6)) # map size

dist.plot(ax=ax, color='lightgrey') # underying map

ax.set_axis_off() # remove x and y axis axis tick marks, labels ...
fig, ax = plt.subplots(figsize=(12, 6))                 # map size

dist.plot(ax=ax, color='lightgrey')                     # underying map

dist.plot(column='totalpopul', ax=ax, cmap='viridis')   # map plotted over the underying map # some values are missing

ax.set_axis_off()                                       # remove x and y axis axis tick marks, labels ...
states = dist.dissolve(by='statename').reset_index() 

states.head(2)
fig, ax = plt.subplots(figsize=(12, 6))                   # map size

states.plot(ax=ax, color='lightgrey')                     # underying map

states.plot(column='totalpopul', ax=ax, cmap='viridis')   # map plotted over the underying map # some values are missing

ax.set_axis_off()                                         # remove x and y axis axis tick marks, labels ...
# plot base map

m = folium.Map(location=[23, 78.9629], # center of the folium map

               tiles='cartodbpositron', # type of map

               min_zoom=4, max_zoom=6, # zoom range

               zoom_start=4) # initial zoom



m
# plot base map

m = folium.Map(location=[23, 78.9629], # center of the folium map

               tiles='cartodbpositron', # type of map

               min_zoom=4, max_zoom=6, # zoom range

               zoom_start=4) # initial zoom



# plot chorpleth over the base map

folium.Choropleth(states,                                # geo data

                  data=states,                           # data

                  key_on='feature.properties.statename', # feature.properties.key

                  columns=['statename', 'totalpopul'],   # [key, value]

                  fill_color='RdPu',                     # cmap

                  line_weight=0.1,                       # line wight (of the border)

                  line_opacity=0.5,                      # line opacity (of the border)

                  legend_name='Population').add_to(m)    # name on the legend color bar



# add layer controls

folium.LayerControl().add_to(m)



m