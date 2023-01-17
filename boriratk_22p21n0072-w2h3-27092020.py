# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
data_airbnb = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data_airbnb.head()
data_airbnb.info()
data_airbnb.describe()
data_airbnb.isnull().sum()
data_airbnb[data_airbnb['reviews_per_month'].isnull()]
data_airbnb = data_airbnb.drop(['id','host_name','last_review'], axis = 1) 



data_airbnb['reviews_per_month'] = data_airbnb['reviews_per_month'].fillna(value=0)
data_airbnb
data_airbnb.isnull().sum()
sns.pairplot(data_airbnb, hue="room_type")
data_airbnb.corr(method='spearman')
plt.figure(figsize=(16, 6))

sns.barplot(data_airbnb['neighbourhood_group'], data_airbnb['price'], hue=data_airbnb['room_type'], ci=None)
plt.figure(figsize=(16, 6))

sns.countplot(data_airbnb['neighbourhood_group'],hue=data_airbnb['room_type'])
plt.figure(figsize=(15, 6))

sns.scatterplot(x=data_airbnb['longitude'], y=data_airbnb['latitude'], hue=data_airbnb['neighbourhood_group'])
plt.figure(figsize=(15, 6))

sns.scatterplot(x=data_airbnb['longitude'], y=data_airbnb['latitude'], hue=data_airbnb['room_type'])
data_for_cluster = data_airbnb[['neighbourhood','number_of_reviews', 'reviews_per_month', 'latitude', 'longitude', 'price']] 

data_for_cluster
data_for_cluster = data_for_cluster.groupby('neighbourhood').agg({'number_of_reviews':'mean','reviews_per_month':'mean', 'latitude': 'mean', 'longitude': 'mean', 'price': 'mean'})

data_for_cluster
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import MinMaxScaler
data_for_cluster.index
fig = plt.figure(figsize=(30,27))

dendogram = dendrogram(linkage(data_for_cluster, method='ward'), leaf_rotation=90, leaf_font_size=12, labels=data_for_cluster.index) 

plt.title("Dendrograms")

plt.xlabel('neighbourhood')

plt.ylabel('Euclidean distances')

plt.show()
fig = plt.figure(figsize=(30,27))

dendogram = dendrogram(linkage(data_for_cluster, method='ward'), leaf_rotation=90, leaf_font_size=12, labels=data_for_cluster.index) 

plt.title("Dendrograms")

plt.xlabel('neighbourhood')

plt.ylabel('Euclidean distances')

plt.axhline(y=800, color='r', linestyle='--')

plt.show()
cluster = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')

cluster_group = cluster.fit_predict(data_for_cluster)

print(cluster_group)
fig = plt.figure(figsize=(20,15))

plt.scatter(data_for_cluster.index, data_for_cluster['number_of_reviews'], c=cluster_group) 

plt.title('cluster = 3')

plt.xlabel('neighbourhood')

plt.xticks(rotation=90)

plt.ylabel('number_of_reviews')

plt.show()
fig = plt.figure(figsize=(20,15))

plt.scatter(data_for_cluster['price'], data_for_cluster['number_of_reviews'], c=cluster_group) 

plt.title('cluster = 3')

plt.xlabel('price')

plt.xticks(rotation=90)

plt.ylabel('number_of_reviews')

plt.show()
fig = plt.figure(figsize=(20,15))

plt.scatter(data_for_cluster.index, data_for_cluster['reviews_per_month'], c=cluster_group) 

plt.title('cluster = 3')

plt.xlabel('neighbourhood')

plt.xticks(rotation=90)

plt.ylabel('reviews_per_month')

plt.show()
fig = plt.figure(figsize=(20,15))

plt.scatter(data_for_cluster['price'], data_for_cluster['reviews_per_month'], c=cluster_group) 

plt.title('cluster = 3')

plt.xlabel('price')

plt.xticks(rotation=90)

plt.ylabel('reviews_per_month')

plt.show()
fig = plt.figure(figsize=(20,15))

plt.scatter(data_for_cluster.index, data_for_cluster['price'], c=cluster_group) 

plt.title('cluster = 3')

plt.xlabel('neighbourhood')

plt.xticks(rotation=90)

plt.ylabel('price')

plt.show()