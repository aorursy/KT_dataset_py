# importing libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from scipy import stats

from scipy.stats import kurtosis

from scipy.stats import skew

from scipy.stats import norm



%matplotlib inline
# Loading our dataset



df_properties = pd.read_csv('../input/dubai-properties-dataset/properties_data.csv')

df_properties.head(3)
# Price analysis



df_properties.price.describe()
fig, ax = plt.subplots(figsize = (8, 8))



sns.distplot(df_properties.price);
print("Skewness: %f" % df_properties.price.skew())

print("Kurtosis: %f" % df_properties.price.kurt())
df_properties_lt5 = df_properties[df_properties.price < 5000001]

df_properties_gt5 = df_properties[df_properties.price > 5000000]
df_properties_lt5.price.describe()
df_properties_gt5.price.describe()
df_properties = df_properties[df_properties.price < 5000001]
fig, ax = plt.subplots(figsize = (8, 8))



sns.distplot(df_properties.price)
print("Skewness: %f" % df_properties.price.skew())

print("Kurtosis: %f" % df_properties.price.kurt())
df_neighborhood_price = df_properties[['price', 'neighborhood']].sort_values(by=['price'], ascending = False)



plt.figure(figsize = (30, 10))

plt.bar(df_neighborhood_price.neighborhood, df_neighborhood_price.price, align='center', alpha=0.5)

plt.xticks(rotation='vertical')

plt.show()
df_neighborhood_price = df_properties[['price_per_sqft', 'neighborhood']].sort_values(by=['price_per_sqft'], ascending = False)



plt.figure(figsize = (30, 10))

plt.bar(df_neighborhood_price.neighborhood, df_neighborhood_price.price_per_sqft, align='center', alpha=0.5)

plt.xticks(rotation='vertical')

plt.show()
#scatter plot grlivarea/saleprice

plt.figure(figsize = (10, 8))



feature = 'size_in_sqft'

plt.scatter(df_properties[feature], df_properties['price'])

plt.xlabel('size_in_sqft')

plt.grid(True)
plt.figure(figsize = (10, 8))



plt.scatter(df_properties['size_in_sqft'], df_properties['no_of_bathrooms'])

plt.xlabel('size_in_sqft')

plt.ylabel('no_of_bathrooms')

plt.grid(True)
plt.figure(figsize = (10, 8))

sns.boxplot(x="no_of_bedrooms", y="no_of_bathrooms", data = df_properties)

plt.show()
df_properties_corr = df_properties.copy()

df_properties_corr.drop(['id', 'latitude', 'longitude'], axis=1, inplace=True)
fig, ax = plt.subplots(figsize=(15, 12))



corr_matrix = df_properties_corr.corr()

sns.heatmap(corr_matrix, annot = False)

plt.show()
from geopy.geocoders import Nominatim

import folium

from folium.plugins import HeatMap



address = 'Dubai, United Arab Emirates'

geolocator = Nominatim(user_agent="data_regress_project")

location = geolocator.geocode(address)

latitude = location.latitude

longitude = location.longitude



data = df_properties[['latitude', 'longitude', 'price']].values



# create map of Dubai using latitude and longitude values

map_dubai_re = folium.Map(location = [latitude, longitude], control_scale=True, zoom_start = 12)



# add markers to map

for lat, lng, neighborhood in zip(df_properties['latitude'], df_properties['longitude'], df_properties['neighborhood']):

    label = '{}'.format(neighborhood)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius = 2,

        popup=label,

        color='b',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=1,

        parse_html=False).add_to(map_dubai_re)  

    

radius = 15

hm = HeatMap(

    data,

    radius=radius,

    blur=30

)

hm.add_to(map_dubai_re)

    

map_dubai_re