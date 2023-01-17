import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import geopandas as gpd

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

import math

import seaborn as sns

import matplotlib.pyplot as plt



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



to_df = pd.read_csv('/kaggle/input/toronto-airbnb-listings/Toronto listings.csv')

tobnb = gpd.GeoDataFrame(to_df, geometry = gpd.points_from_xy(to_df.longitude,to_df.latitude))



#To blur the PII from the dataset

tobnb.drop(['host_id','host_name'],axis=1,inplace=True)

tobnb.crs = {'init': 'epsg:4326'}
tobnb.head(10)

#How many row items does this dataset have

len(tobnb)
# The most pricey airbnb listing in Toronto GTA

tobnb.loc[tobnb['price'].idxmax()]

tobnb.sort_values('price',ascending=False).iloc[:5,]
# The cheapest airbnb listing in Toronto GTA

tobnb.loc[tobnb['price'].idxmin()]

tobnb.sort_values('price',ascending=True).iloc[:10,]
##Create a visualization for the price distribution

sns.distplot(tobnb.price)

plt.xlim(0,500)

tobnb.price.mean()
#To look at the neighbourhoods with the top number of listings

top_neighbourhood = tobnb.neighbourhood.value_counts()

top_neighbourhood
#Create the distribution of prices of Toronto Airbnb listings

sns.set(style = 'darkgrid')

neighourhood_count = sns.countplot(y='neighbourhood',data=tobnb,palette='Greens_d',order=tobnb.neighbourhood.value_counts().iloc[:10].index)

topdtneighbourhoods = ['Waterfront Communities-The Island','Annex','Church-Yonge Corridor','Bay Street Corridor','Kensington-Chinatown']

sub1 = tobnb[tobnb.neighbourhood.isin(topdtneighbourhoods)]

avg_price = sub1.groupby('neighbourhood')['price'].mean()

avg_price
fig, axes = plt.subplots(1,1,figsize=(18.5, 6))

sns.distplot(tobnb['availability_365'], rug=False, kde=False, color="blue", ax=axes)

axes.set_xlabel('availability_365')

axes.set_xlim(0, 365)
d_map = tobnb.plot(figsize=(20,20))

#an unoptimized map where you can roughly see Toronto GTA's shape
def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='300px')
# Create a map

im = folium.Map(location=[43.6532,-79.3832], tiles='openstreetmap', zoom_start=13)

embed_map(im, 'im.html')
#A more human readable way than plotting every single occurence 

im3 = folium.Map(location=[43.6532,-79.3832], tiles='cartodbpositron', zoom_start=13)

mc = MarkerCluster()

for idx, row in tobnb.iterrows():

    if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):

        mc.add_child(Marker([row['latitude'], row['longitude']]))

im3.add_child(mc)

embed_map(im3,'im3.html')
#The last visualization is a heatmap

im4 = folium.Map(location=[43.6532,-79.3832], tiles='cartodbpositron', zoom_start=11)

HeatMap(data=tobnb[['latitude','longitude']],radius=10).add_to(im4)

embed_map(im4,'im4.html')