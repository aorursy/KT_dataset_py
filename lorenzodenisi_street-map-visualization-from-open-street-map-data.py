import geojson

import sys, os, numpy as np

import matplotlib.pyplot as plt

from matplotlib import font_manager as rcParams

from shapely.geometry import Polygon, LineString

from pyproj import Proj, transform



from datashader.utils import lnglat_to_meters as webm
# geojson data can be taken from overpass turbo (https://overpass-turbo.eu/)

with open("../input/turin.geojson") as file:

    data = geojson.load(file)
roads = {}

rails = {}



for path in data["features"]:

    # street data

    if "highway" in path["properties"].keys():

        if path["properties"]["highway"] not in roads.keys():

            roads[path["properties"]["highway"]] = []

            

        if path["geometry"]["type"]=="Polygon":

            cart = [[x, y] for x, y in path["geometry"]["coordinates"][0]]

            roads[path["properties"]["highway"]].append(Polygon(cart))

                     

            

        elif path["geometry"]["type"]=="LineString":

            cart = [[x, y] for x, y in path["geometry"]["coordinates"]]

            roads[path["properties"]["highway"]].append(LineString(cart))

                      

    # railway data

    elif "railway" in path["properties"].keys():

        if path["properties"]["railway"] not in rails.keys():

            rails[path["properties"]["railway"]] = []

            

        if path["geometry"]["type"]=="Polygon":

            cart = [[x, y] for x, y in path["geometry"]["coordinates"][0]]

            rails[path["properties"]["railway"]].append(Polygon(cart))

            

        elif path["geometry"]["type"]=="LineString":

            cart = [[x, y] for x, y in path["geometry"]["coordinates"]]

            rails[path["properties"]["railway"]].append(LineString(cart))
def plot_paths(ax, paths, color, width, linestyle):

    for path in paths:

        if isinstance(path, Polygon):

            x, y = path.exterior.xy   

        elif isinstance(path, LineString):

            x, y = path.xy

        else:

            continue

        

        

        mercator = webm(list(x), list(y))

        

        ax.plot(mercator[0], mercator[1], color=color, linewidth=width, linestyle=linestyle, solid_capstyle='round')    
roads.keys()
scale=1

fig, ax = plt.subplots(figsize=(10*scale, 10*scale))



road_color = '#FCEF3C'

rail_color = '#FCEF3C'

bg_color = '#0C3C7C'





plot_paths(ax, roads["motorway"], road_color, 0.8*scale, "-")

plot_paths(ax, roads["motorway_link"], road_color, 0.8*scale, "-")

plot_paths(ax, roads["trunk"], road_color, 0.8*scale, "-")

plot_paths(ax, roads["trunk_link"], road_color, 0.8*scale, "-")



plot_paths(ax, roads["primary"], road_color, 0.6*scale, "-")

plot_paths(ax, roads["primary_link"], road_color, 0.6*scale, "-")



plot_paths(ax, roads["secondary"], road_color, 0.4*scale, "-")

plot_paths(ax, roads["secondary_link"], road_color, 0.4*scale, "-")

plot_paths(ax, roads["tertiary"], road_color, 0.4*scale, "-")

plot_paths(ax, roads["tertiary_link"], road_color, 0.4*scale, "-")



plot_paths(ax, roads["residential"], road_color, .3*scale, "-")

plot_paths(ax, roads["track"], road_color, .3*scale, "-")

plot_paths(ax, rails["rail"], rail_color, 0.15*scale, '-.')



plot_paths(ax, roads["pedestrian"], road_color, .2*scale, "-")

plot_paths(ax, roads["path"], road_color, .2*scale, "-")

plot_paths(ax, roads["service"], road_color, .2*scale, "-")

plot_paths(ax, roads["unclassified"], road_color, .2*scale, "-")

plot_paths(ax, roads["living_street"], road_color, .2*scale, "-")



ax.set_facecolor(bg_color)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.get_xaxis().set_ticks([])

ax.get_yaxis().set_ticks([])

#plt.grid(linewidth=1,color='#5c5c5c')

plt.savefig("aaa.png")

plt.show()