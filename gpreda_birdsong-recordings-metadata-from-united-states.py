!pip install geojsoncontour
import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import folium

from folium.plugins import HeatMap, HeatMapWithTime

from scipy.interpolate import griddata

import geojsoncontour

import scipy as sp

import scipy.ndimage

import branca

from folium import plugins

%matplotlib inline
birds_df = pd.read_csv("/kaggle/input/bird-songs-recordings-from-united-states/birds_united_states.csv")
birds_df.head()
birds_df.info()
birds_df.describe()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(birds_df)
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
unique_values(birds_df)
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set2')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
plot_count("sp", "Species", birds_df, size=4)
plot_count("ssp", "Subspecies", birds_df, size=4)
plot_count("en", "Species (English name)", birds_df, size=4)
plot_count("gen", "Latin name (for gen)", birds_df, size=4)
plot_count("rec", "Recorder person", birds_df, size=4)
plot_count("bird-seen", "`was the bird seen?`", birds_df, size=2)
plot_count("playback-used", "`was playback used?`", birds_df, size=2)
aggregated_df = birds_df.groupby(["lat", "lng"])["id"].count().reset_index()

aggregated_df.columns = ['lat', 'lng', 'count']
m = folium.Map(location=[37, -95], zoom_start=4)

max_val = max(aggregated_df['count'])

HeatMap(data=aggregated_df[['lat', 'lng', 'count']],\

        radius=15, max_zoom=12).add_to(m)

m
def alt_conv(x):

    

    try:

        x = float(x)

    except:

        x = 0.0

    return x
filtered_df = birds_df[["lat", "lng", "alt"]]

filtered_df.columns = ['lat', 'lng', 'altitude']

filtered_df['altitude']  = filtered_df['altitude'].apply(lambda x: alt_conv(x))

filtered_df = filtered_df.dropna()



# focus on only US mainland, and excluding Alaska

filtered_df = filtered_df.loc[(filtered_df.lat<50) & (filtered_df.lat>25)&(filtered_df.lng>-125)&(filtered_df.lng<-70)]
# define the colors for the elevation (altitude) map

colors = ['blue','royalblue', 'navy','pink',  'mediumpurple',  'darkorchid',  'plum',  'm', 'mediumvioletred', 'palevioletred', 'crimson',

         'magenta','pink','red','yellow','orange', 'brown','green', 'darkgreen']

levels = len(colors)



# create arrays for the x,y,z grid

x = np.asarray(filtered_df.lng.tolist())

y = np.asarray(filtered_df.lat.tolist())

z = np.asarray(filtered_df.altitude.tolist()) 



vmin = filtered_df['altitude'].min() 

vmax = filtered_df['altitude'].max()



# create a grid

x_arr          = np.linspace(np.min(x), np.max(x), 5000)

y_arr          = np.linspace(np.min(y), np.max(y), 5000)

x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

 



z_mesh = griddata((x, y), z, (x_mesh, y_mesh), method='linear')

 

# use Gaussian filter to smoothen the contour

sigma = [5, 5]

z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')

 

# create the contour

contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels, alpha=0.5, colors=colors, linestyles='None', vmin=vmin, vmax=vmax)
# convert matplotlib contourf to geojson

geojson = geojsoncontour.contourf_to_geojson(

    contourf=contourf,

    min_angle_deg=3.0,

    ndigits=5,

    stroke_width=1,

    fill_opacity=0.1)



cm  = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)



# set up the map placeholdder

geomap_elevation = folium.Map([filtered_df.lat.mean(), filtered_df.lng.mean()], zoom_start=4, tiles="OpenStreetMap")

# plot the contour on Folium map

folium.GeoJson(

    geojson,

    style_function=lambda x: {

        'color':     x['properties']['stroke'],

        'weight':    x['properties']['stroke-width'],

        'fillColor': x['properties']['fill'],

        'opacity':   0.5,

    }).add_to(geomap_elevation)

 

# add the colormap to the folium map for legend

cm.caption = 'Elevation'

geomap_elevation.add_child(cm)

 

# add the legend to the map

plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap_elevation)

geomap_elevation