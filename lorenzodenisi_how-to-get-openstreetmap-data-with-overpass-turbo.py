import geojson #used to handle geojson files

import matplotlib.pyplot as plt #used to plot the final graphs

from shapely.geometry import LineString #used to handle shape and path objects

import pandas as pd 

import seaborn as sns

import numpy as np



from datashader.utils import lnglat_to_meters as webm
with open("../input/osm-overpass-tutorial-dataset/pubs.geojson") as file:

    data = geojson.load(file)
data.keys()
data.features[0].keys()
nodes = []

for node in data.features:

    nodes.append(node.geometry.coordinates)
from matplotlib.colors import LogNorm



df = pd.DataFrame(nodes)

df.columns = ["Longitude", "Latitude"]

df.Longitude, df.Latitude = webm(df.Longitude, df.Latitude)

df["Lat_binned"] = pd.cut(df.Latitude, 150)

df["Lon_binned"] = pd.cut(df.Longitude, 150)



df = df.pivot_table(

        values='Latitude', 

        index='Lat_binned', 

        columns='Lon_binned', 

        aggfunc=np.size)

df = df[::-1] #reverse latitude values

df = df.fillna(1) #pivoting produces nans that needs to be converted to values to be displayed (I cannot fill with zero because the color scale is logarithmic)



fig, ax = plt.subplots(figsize=(9, 12.24))

log_norm = LogNorm(vmin=df.min().min(), vmax=df.max().max())

sns.heatmap(df, norm = log_norm, ax = ax)

plt.axis("off");
with open("../input/osm-overpass-tutorial-dataset/roads.geojson") as file:

    data = geojson.load(file)



data.keys()
data.features[0].properties.keys()
roads = {}

rails = {}

for path in data.features:

        # street data

    if "highway" in path.properties.keys():

        if path["properties"]["highway"] not in roads.keys():

            roads[path["properties"]["highway"]] = []

        

        '''

        if path["geometry"]["type"]=="Polygon":

            cart = [[x, y] for x, y in path["geometry"]["coordinates"][0]]

            roads[path["properties"]["highway"]].append(Polygon(cart))

                     

        '''    

        if path["geometry"]["type"]=="LineString":

            cart = [[x, y] for x, y in path["geometry"]["coordinates"]]

            roads[path["properties"]["highway"]].append(LineString(cart))

                      

    # railway data

    elif "railway" in path["properties"].keys():

        if path["properties"]["railway"] not in rails.keys():

            rails[path["properties"]["railway"]] = []

         

        '''

        if path["geometry"]["type"]=="Polygon":

            cart = [[x, y] for x, y in path["geometry"]["coordinates"][0]]

            rails[path["properties"]["railway"]].append(Polygon(cart))

        '''

        

        if path["geometry"]["type"]=="LineString":

            cart = [[x, y] for x, y in path["geometry"]["coordinates"]]

            rails[path["properties"]["railway"]].append(LineString(cart))
str(roads["primary"][0])
def plot_paths(ax, paths, color, width, linestyle):

    for path in paths:

        '''

        if isinstance(path, Polygon):

            x, y = path.exterior.xy   

        '''

        

        if isinstance(path, LineString):

            x, y = path.xy

        else:

            continue

        

        mercator = webm(list(x), list(y))

        

        

        ax.plot(mercator[0], mercator[1], color=color, linewidth=width, linestyle=linestyle, solid_capstyle='round')
roads.keys()
rails.keys()
fig, ax = plt.subplots(figsize=(10, 10))



bg = '#011654'

rail_color = '#CF142B'

street_color = '#FFFFFF'



plot_paths(ax, roads["motorway"], street_color, 0.8, "-")

plot_paths(ax, roads["motorway_link"], street_color, 0.8, "-")

plot_paths(ax, roads["trunk"], street_color, 0.8, "-")

plot_paths(ax, roads["trunk_link"], street_color, 0.8, "-")



plot_paths(ax, roads["primary"], street_color, 0.6, "-")

plot_paths(ax, roads["primary_link"], street_color, 0.6, "-")



plot_paths(ax, roads["secondary"], street_color, 0.4, "-")

plot_paths(ax, roads["secondary_link"], street_color, 0.4, "-")



plot_paths(ax, rails["rail"], rail_color, 0.6, '-.')



ax.set_facecolor(bg)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.get_xaxis().set_ticks([])

ax.get_yaxis().set_ticks([])



plt.show()