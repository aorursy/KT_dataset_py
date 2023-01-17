import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
dataset.drop(["last_review", "reviews_per_month"], axis = 1, inplace = True)

dataset.head()
data =dataset[['neighbourhood','room_type','price','calculated_host_listings_count']]

data
data['neighbourhood'].value_counts()
data['room_type'].value_counts()
dataWithoutLabels = data.drop(["room_type","neighbourhood"], axis = 1)

dataWithoutLabels.head()
dataWithoutLabels.info()
sns.pairplot(data.loc[0:1000,['price','calculated_host_listings_count','room_type']], hue = "room_type", height = 4)

plt.show()
plt.figure(figsize = (10, 10))

plt.scatter(dataWithoutLabels["price"], dataWithoutLabels["calculated_host_listings_count"])

plt.xlabel('price')

plt.ylabel('calculated_host_listings_count')

plt.show()
dataWithoutLabels_1 = dataWithoutLabels.loc[1:4000,['price','calculated_host_listings_count']]

dataWithoutLabels_1
grouped_room = data.groupby('room_type').mean().reset_index().loc[:,['room_type','price']]

list_grouped_room_type = [ [price] for price in grouped_room.price.to_list()]

grouped_room.head()
grouped_neighbourhood = data.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','calculated_host_listings_count']]

list_grouped_neighbourhood = [ [calculated_host_listings_count] for calculated_host_listings_count in grouped_neighbourhood.calculated_host_listings_count.to_list()]

grouped_neighbourhood.head()
from scipy.cluster.hierarchy import linkage,dendrogram



M = linkage(list_grouped_room_type, 'complete')

N = linkage(list_grouped_neighbourhood, 'complete')
plt.figure(figsize = (10, 5))

dendrogram(M, leaf_rotation = 90)

plt.title('Dendrogram')

plt.show()
plt.figure(figsize = (20, 25))

dendrogram(N, leaf_rotation = 90)

plt.title('Dendrogram')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")

cluster = hc.fit_predict(dataWithoutLabels_1)

dataWithoutLabels_1["label"] = cluster
dataWithoutLabels_1.label.value_counts()
plt.figure(figsize = (15, 10))

plt.scatter(dataWithoutLabels_1["price"][dataWithoutLabels_1.label == 0], dataWithoutLabels_1["calculated_host_listings_count"][dataWithoutLabels_1.label == 0], color = "red")

plt.scatter(dataWithoutLabels_1["price"][dataWithoutLabels_1.label == 1], dataWithoutLabels_1["calculated_host_listings_count"][dataWithoutLabels_1.label == 1], color = "green")

plt.scatter(dataWithoutLabels_1["price"][dataWithoutLabels_1.label == 2], dataWithoutLabels_1["calculated_host_listings_count"][dataWithoutLabels_1.label == 2], color = "blue")





plt.xlabel("price")

plt.ylabel("calculated_host_listings_count")

plt.show()