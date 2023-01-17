import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import seaborn as sns

import pandasql as ps

sns.set()
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head(3)
print(data.shape)
data.dtypes
numeric_nyc_airbnb=data._get_numeric_data().columns

print(numeric_nyc_airbnb)
categorical_nyc_airbnb=set(data.columns) - set(numeric_nyc_airbnb)

print(categorical_nyc_airbnb)
data.isnull().sum()
data.drop(['id','last_review'], axis=1, inplace=True)
data.isnull().sum()
data.fillna({'reviews_per_month':0}, inplace=True)
data.isnull().sum()
data.neighbourhood_group.unique()
plt.figure(figsize=(8,4))

sns.countplot(data.sort_values('neighbourhood_group').neighbourhood_group,palette='Set3',alpha=0.8)

plt.title('Neighbourhood wise Airbnb listings in NYC')

plt.xlabel('Neighbourhood name')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(10,8))

sns.scatterplot(x='longitude', y='latitude',hue='neighbourhood_group',s=20,palette='CMRmap_r', data=data)
data.room_type.unique()
plt.figure(figsize=(8,4))

sns.countplot(data.sort_values('room_type').room_type,palette='husl', hue=data.room_type,alpha=0.8)

plt.title('Room Type distribution in NYC')

plt.xlabel('Room Type')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(10,8))

sns.scatterplot(x='longitude', y='latitude',hue='room_type',palette='afmhot',s=20, data=data)
data.neighbourhood.unique()
plt.figure(figsize=(15,6))

sns.countplot(data.sort_values('neighbourhood').neighbourhood, order=data['neighbourhood'].value_counts().iloc[:10].index ,palette='YlGnBu_r',alpha=0.8)

plt.title('Neighbourhood wise Airbnb listings in NYC')

plt.xlabel('Neighbourhood name')

plt.ylabel('Count')

plt.show()
name_count=data.name.value_counts().head(10)

name_count
plt.grid(b=None)

data['name'].value_counts()[:10].sort_values().plot(kind='barh', color='green', linestyle='-.')
top_host=data.host_id.value_counts().head(10)

top_host
plt.grid(b=None)

data['host_id'].value_counts()[:10].sort_values().plot(kind='barh', color='Brown', linestyle='-.')
plt.figure(figsize=(8,4))

sns.countplot(data.sort_values('neighbourhood_group').neighbourhood_group,hue=data.room_type, palette='husl',alpha=0.8)

plt.title('Room Type distribution in NYC')

plt.xlabel('Room Type')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(data.price, color='g')

plt.title('Distribution of price')

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(np.log1p(data['price']),color='g')

plt.title('Distribution of log of price')

plt.show()
plt.figure(figsize=(8,8))

sns.distplot(data['minimum_nights'], color='b')

plt.xlabel('Minimum Nights')

plt.xlim(0,300)

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(np.log1p(data['minimum_nights']),color='b')

plt.title('Distribution of log of minimum nights')

plt.show()
plt.figure(figsize=(8,8))

sns.distplot(data['number_of_reviews'], color='brown')

plt.xlabel('Number of reviews')

plt.xlim(0,300)

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(np.log1p(data['number_of_reviews']),color='brown')

plt.title('Distribution of log of number of reviews')

plt.show()
plt.figure(figsize=(8,8))

sns.distplot(data['reviews_per_month'], color='brown')

plt.xlabel('Reviews per month')

plt.xlim(0,300)

plt.show()
f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap="BuPu")

plt.show()