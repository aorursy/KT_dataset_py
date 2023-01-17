import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

#For geoplots

import geopandas as gpd

from shapely.geometry import Point, Polygon

import descartes
#Reading the dataset.

data = pd.read_csv(r'/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#Exploring the top 5 observations.

data.head()
#describing the datasets.

data.describe()
#null or missing values in the dataset.

data.isnull().sum()
#checking the assumption -> 0 reviews will have missing values in last_review and reviews_per_month columns.

assumption_test = data.loc[(data.last_review.isnull()) & (data.reviews_per_month.isnull()), ['number_of_reviews',  'last_reviews', 'reviews_per_month']]

assumption_test.head()
assumption_test.shape
#filling the missing values in reviews_per_month with 0.

data.reviews_per_month.fillna(0, inplace=True)
#Checking if the changes made are reflected.

data.isnull().sum()
#checking the to 5 neighborhood where the properties are listed most.

top_5_neighborhoods = data.neighbourhood.value_counts().head(5)

print(top_5_neighborhoods)



#plotting 

plt.figure(figsize=(8,5))

top_5_neighborhoods.plot.bar()

plt.xlabel('Neighborhoods')

plt.ylabel('Listed Property Count')

plt.title('Count of properties in a neighborhood')

plt.show() #optional
#checking the to 5 neighborhood groups where the properties are listed most.

top_5_neighborhood_group = data.neighbourhood_group.value_counts()

print(top_5_neighborhood_group)



#plotting 

plt.figure(figsize=(8,5))

top_5_neighborhood_group.plot.bar()

plt.xlabel('Neighborhood Groups')

plt.ylabel('Listed Property Count')

plt.title('Count of properties in a neighborhood group')

plt.show() #optional
#number of rooms_type provided by the hosts

print(data.room_type.value_counts())

sns.countplot(data.room_type)
#Lets check the distribution of the price of the properties.

sns.distplot(data.price, bins=50)
#Looking into the properties having 0 Price

free_properties = data.loc[data.price <= 0]

print('Shape of the data:', free_properties.shape)

free_properties.head()
#minimum number of nights allowed by the host.

sns.distplot(data.minimum_nights, bins=10)
#properties recieving highest reviews.

highest_reviews = data.sort_values(by='number_of_reviews', ascending=False)

highest_reviews.head()
#host having highest amount of properties listed.

highest_props_host = data.groupby(['host_id', 'host_name'])['host_id'].count().sort_values(ascending=False)[:10]

highest_props_host.plot.bar(figsize=(10,5))

plt.xlabel('Hosts')

plt.ylabel('Properties Listed')

plt.title('Hosts having highest amount of properties listed');
#neighborhood group based on the latitude and longitude

plt.figure(figsize=(10,8))

sns.scatterplot(data.latitude,data.longitude, hue='neighbourhood_group', data=data)
# #Properties in the neighbourhood with most reviews. 

# plt.figure(figsize=(10,8))

# sns.scatterplot('latitude', 'longitude', hue='neighbourhood_group', data=highest_reviews.head(10))