# Import the required packages

import sklearn

import pandas as pd

import matplotlib.pyplot as plt

import geopandas as gpd
import shapely

import numpy as np

import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

from sklearn.neighbors import KNeighborsClassifier
import folium
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
## Checking the data types

df.dtypes
df.isnull().sum()
## id, last_review, reviews_per_month columns can be dropped



df.drop(['id','host_name','last_review', 'reviews_per_month'], axis=1, inplace=True)
# Checking the categories of each column 

df['room_type'].unique()

df['neighbourhood_group'].unique()

df['neighbourhood'].unique()
df['price'].describe()
df['minimum_nights'].describe()
df['number_of_reviews'].describe()
## Seperating the first and the last word from the name of the property



df['last'] = df["name"].apply(lambda x: str(x).split()[-1])

df['first'] = df["name"].apply(lambda x: str(x).split()[0])
## Areas with most rooms available for 365 days



a365 = df[df['availability_365'] == 365]

a365 = a365['neighbourhood'].value_counts()

(a365.head(n = 20)).plot.bar(figsize =(20, 20), title = "Localities with most number of rooms available for 365 days")
# Neighborhoods with most places for rent

count = df['neighbourhood'].value_counts()

(count.head(n = 20)).plot.bar(figsize =(20, 20), title = "Localities with most number of rooms available")
# Number of reviews per Neighbourhood

pd.DataFrame(df[['neighbourhood_group', 'number_of_reviews']].groupby(['neighbourhood_group']).agg(['count'])).plot.bar(figsize = (20,20), title = "Number of reviews per locality")
# Costliest Neighborhoods(Groups)



a = pd.DataFrame(df[['neighbourhood_group', 'price']].groupby(['neighbourhood_group']).agg(['mean']))



a.plot.bar(figsize = (20,20), title = "Mean price of each Locality")
## Which Adjectives(first word) received the most number of reviews 

word = pd.DataFrame(df[['first', 'number_of_reviews']].groupby(['first']).agg(['count']))

word.columns=['totalreviews']

adjective = word.nlargest(20, ['totalreviews']) 

adjective.plot.bar(figsize = (20,20), title = "Top Adjectives according to the number reviews")
## Top 20 Costliest Neighbourhoods(According to mean price)

word = pd.DataFrame(df[['neighbourhood', 'price']].groupby(['neighbourhood']).agg(['mean']))

word.columns=['totalprice']

adjective = word.nlargest(20, ['totalprice']) 

adjective.plot.bar(figsize = (20,20), title = "Top Locations according to Mean Price")
# Does the last word tell us anything



word = pd.DataFrame(df[['last', 'number_of_reviews']].groupby(['last']).agg(['count']))

word.columns=['totalreviews']

adjective = word.nlargest(20, ['totalreviews']) 

adjective.plot.bar(figsize = (20,20), title = "Which last words gained most reviews")



## Owners often used the locality names to gain cutomers and for better reviews; for eg. "Affordable room in Bushwick/East Williamsburg" or "Trendy duplex in the very heart of Hell's Kitchen" 
# Plotting Locations

m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start = 10, tiles = 'Open Street Map')





for _, row in df.iterrows():

    folium.CircleMarker(location=[row.latitude, row.longitude],

                       radius = 4, 

                       popup = row.name,

                       color = '#1787FE',

                       fill = True,

                        fill_color = '#1787FE').add_to(m)

    

m    
## Let us convert the object and string data types to first categorical type and then to integer for clustering



df['neighbourhood_group'] = df['neighbourhood_group'].astype('category')

df['neighbourhood_group'] = df['neighbourhood_group'].cat.codes



df['room_type'] = df['room_type'].astype('category')

df['room_type'] = df['room_type'].cat.codes



df['neighbourhood'] = df['neighbourhood'].astype('category')

df['neighbourhood'] = df['neighbourhood'].cat.codes



df['last'] = df['last'].astype('category')

df['last'] = df['last'].cat.codes



df['first'] = df['first'].astype('category')

df['first'] = df['first'].cat.codes
X = np.array(df[['longitude', 'latitude', 'host_id', 'neighbourhood', 'neighbourhood_group', 'room_type', 'number_of_reviews', 'price', 'minimum_nights', 'calculated_host_listings_count', 'availability_365', 'last', 'first']], dtype='float64')

## Making a column for clusters



X = np.array(df[['longitude', 'latitude', 'host_id', 'neighbourhood', 'neighbourhood_group', 'room_type', 'number_of_reviews', 'price', 'minimum_nights', 'calculated_host_listings_count', 'availability_365', 'last', 'first']])



k = 70

model = KMeans(n_clusters=k, random_state=17).fit(X)

class_predictions = model.predict(X)

df[f'Cluster_'] = class_predictions
## Check the clusters column

df