import numpy as np

import pandas as pd
import seaborn as sns
datasetListingsDetailed=pd.read_csv("/kaggle/input/madrid-airbnb-data/listings_detailed.csv")



datasetListingsDetailed.head(1)
#I made this to see every column of datasetListingsDetailed

for col in datasetListingsDetailed.columns:

    print (col)
datasetListings=pd.read_csv("/kaggle/input/madrid-airbnb-data/listings.csv")

datasetListings.head(1)
import geopandas as gpd

import folium

from folium import Marker

import matplotlib.pyplot
mapMadrid=gpd.read_file("/kaggle/input/madrid-airbnb-data/neighbourhoods.geojson")
mapMadrid
mapMadrid.plot(figsize=(4,4))

#we are going to create a geodataframe

gdf=gpd.GeoDataFrame.from_features(mapMadrid)

gdf['lon']=gdf['geometry'].centroid.x

gdf['lat']=gdf['geometry'].centroid.y

# set the coordinate reference system (CRS) to EPSG 4326

gdf.crs = {'init': 'epsg:4326'}

gdf.head()
gdf.describe()
ax=gdf.plot()

# we create the map 

m = folium.Map(location=[40.424839,

                         -3.682028], tiles='cartodbpositron', zoom_start=12)



# we add the neighbourhoods to the map

for idx, row in gdf.iterrows():

    Marker([row['lat'], row['lon']], popup=row['neighbourhood']).add_to(m)



# display the map

m
centro = datasetListings[((datasetListings.neighbourhood_group == 'Centro'))]

centro.head()
centro.describe()
from folium.plugins import HeatMap, MarkerCluster
#in latitude and longtiude we use the mean latitude and mean longitude of centro.describe()

m_2 = folium.Map(location=[40.416663,

                         -3.703772], tiles='cartodbpositron', zoom_start=15)



# we add a heatmap to the base map

HeatMap(data=centro[['latitude', 'longitude']], radius=10).add_to(m_2)



# display the map

m_2
from folium import Circle
sorted(list(set(datasetListingsDetailed['review_scores_rating'].values)))
#There are a lot of nan values, I'm going to fill these ones using the mean

for column in ['review_scores_rating']:

    datasetListingsDetailed[column].fillna(datasetListingsDetailed[column].mode()[0], inplace=True)
sorted(list(set(datasetListingsDetailed['review_scores_rating'].values)))
malasaña=datasetListingsDetailed[((datasetListingsDetailed.neighbourhood == 'Malasaña'))]
malasaña.head()
print(malasaña.latitude.mean())

malasaña.longitude.mean()
m_3 = folium.Map(location=[40.424228, -3.7051433], tiles='cartodbpositron', zoom_start=30)





def color_producer(val):

    if val < 95: #if the review_score_rating of the place is below or equal to 95

        return 'forestgreen'

    else:

        return 'darkred'
for i in range(0,len(malasaña)):

    Circle(

        location=[malasaña.iloc[i]['latitude'], malasaña.iloc[i]['longitude']],

        radius=20,

        color=color_producer(malasaña.iloc[i]['review_scores_rating'])).add_to(m_3)
m_3
m_4 = folium.Map(location=[40.424228, -3.7051433], tiles='cartodbpositron', zoom_start=25)

def color_producer(val):

    if val >=5: #if the airbnb has 5 or more beds 

        return 'darkred'

    else:

        return 'blue'
for i in range(0,len(malasaña)):

    Circle(

        location=[malasaña.iloc[i]['latitude'], malasaña.iloc[i]['longitude']],

        radius=20,

        color=color_producer(malasaña.iloc[i]['beds'])).add_to(m_4)
m_4