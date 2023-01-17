import numpy as np

import pandas as pd

import geopandas as gpd



import folium

from folium import Choropleth

from folium.plugins import HeatMap



import matplotlib.pyplot as plt

import seaborn as sns
# Read the data in

df_airbnb = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

# Have a look at the first 5 entries

df_airbnb.head(5)
# Print some stats about the data

print('There are {} listings in the dataset'.format(len(df_airbnb)))

print('The dataset has {} columns with the following titles:\n'.format(len(df_airbnb.columns)))

for col in df_airbnb.columns:

    print(col)
# Check which columns have missing data

df_airbnb.isnull().sum()
# Drop the 3 columns and have a look at the new dataset

df_airbnb.drop(['id','host_name','last_review'], axis=1, inplace=True)

df_airbnb.head(5)
# Fill the missing reviews per month with zero

df_airbnb.fillna({'reviews_per_month': 0}, inplace=True)

df_airbnb.head(5)
# One last check to make sure there is no missing data

df_airbnb.isnull().sum()
gdf_nyc_boundaries = gpd.read_file("../input/nyc-neighbourhood-boundaries/geo_export_9f5919c7-b377-48b3-bc3f-c45b49cba31a.shp")

gdf_nyc_boundaries.set_index('ntacode', inplace=True)

gdf_nyc_boundaries.head(5)
count_data_df = df_airbnb[['neighbourhood']].neighbourhood.value_counts()

gdf_nyc_boundaries["listing_count"] = 0



for ii in range(len(count_data_df)):

    key = count_data_df.keys()[ii]

    

    index = gdf_nyc_boundaries.index[gdf_nyc_boundaries['ntaname'] == key]

    if len(index) > 0:

        gdf_nyc_boundaries.loc[index[0], ('listing_count')] = count_data_df[ii]



gdf_nyc_boundaries.head()
# This function is used to help plot the maps in the notebook

def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
listing_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)

Choropleth(geo_data=gdf_nyc_boundaries['geometry'].__geo_interface__,

           key_on="feature.id",

           data=gdf_nyc_boundaries['listing_count'],

           fill_color='YlOrRd',

           legend_name='NYC Neighbourhoods'

          ).add_to(listing_map)

embed_map(listing_map, 'listing_map.html')
df_airbnb.price.describe()
# We will use the 25 and 75 percentile for the plot

def color_producer(val):

    if val < 69:

        return 'forestgreen'

    if val < 175:

        return 'darkorange'

    else:

        return 'darkred'

    

price_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)

    

for i in range(0, len(df_airbnb)):

    folium.Circle(

        location=[df_airbnb.iloc[i]['latitude'], df_airbnb.iloc[i]['longitude']],

        radius=25,

        color=color_producer(df_airbnb.iloc[i]['price'])

    ).add_to(price_map)

embed_map(price_map, 'price_map.html')
df_airbnb_top_reviewed = df_airbnb.nlargest(100, 'number_of_reviews')

df_airbnb_top_reviewed[['number_of_reviews', 'price']].describe()
def color_producer(val):

    if val < 348:

        return 'forestgreen'

    if val < 427:

        return 'darkorange'

    else:

        return 'darkred'

    

review_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)

    

for i in range(0, len(df_airbnb_top_reviewed)):

    folium.Circle(

        location=[df_airbnb_top_reviewed.iloc[i]['latitude'], df_airbnb_top_reviewed.iloc[i]['longitude']],

        radius=25,

        color=color_producer(df_airbnb_top_reviewed.iloc[i]['number_of_reviews'])

    ).add_to(review_map)

embed_map(review_map, 'review_map.html')
df_airbnb_rooms = df_airbnb.room_type.unique()



# Print some stats about the types of rooms

print('There are {} types of rooms\n'.format(len(df_airbnb_rooms)))

for ii in range(len(df_airbnb_rooms)):

    print(df_airbnb_rooms[ii])
# Get the average price and number of reviews per room type

df_airbnb_rooms_private = df_airbnb.loc[df_airbnb['room_type'] == 'Private room']

df_airbnb_rooms_entire = df_airbnb.loc[df_airbnb['room_type'] == 'Entire home/apt']

df_airbnb_rooms_shared = df_airbnb.loc[df_airbnb['room_type'] == 'Shared room']



print('Private Room Stats')

df_airbnb_rooms_private[['price', 'number_of_reviews']].describe()
print('Entire Home/Apt Stats')

df_airbnb_rooms_entire[['price', 'number_of_reviews']].describe()
print('Shared Room Stats')

df_airbnb_rooms_shared[['price', 'number_of_reviews']].describe()
def color_producer(val):

    if val < 'Private room':

        return 'forestgreen'

    if val < 'Shared room':

        return 'darkorange'

    else:

        return 'darkred'

    

types_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)

    

for i in range(0, len(df_airbnb)):

    folium.Circle(

        location=[df_airbnb.iloc[i]['latitude'], df_airbnb.iloc[i]['longitude']],

        radius=25,

        color=color_producer(df_airbnb.iloc[i]['room_type'])

    ).add_to(types_map)

embed_map(types_map, 'types_map.html')
df_airbnb.availability_365.describe()
def color_producer(val):

    if val > 227:

        return 'forestgreen'

    if val > 0:

        return 'darkorange'

    else:

        return 'darkred'

    

availability_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)

    

for i in range(0, len(df_airbnb)):

    folium.Circle(

        location=[df_airbnb.iloc[i]['latitude'], df_airbnb.iloc[i]['longitude']],

        radius=25,

        color=color_producer(df_airbnb.iloc[i]['availability_365'])

    ).add_to(review_map)

embed_map(availability_map, 'availability_map.html')