# This is my first version of Kritsatorn

# Just showing Diagram 

# i wil improve my skill in next week

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.plot(x="longitude", y="latitude", style=".", figsize=(20, 20))

plt.title("Map")

plt.ylabel("latitude")

img = plt.imread("/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png", 0)

plt.imshow(img, extent=[-74.25, -73.685, 40.49, 40.925])

plt.show()
int_col = ['id','host_id','latitude','longitude','price','minimum_nights','number_of_reviews' ,'calculated_host_listings_count','availability_365']
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch

import sklearn.datasets
dendrogram = sch.dendrogram(sch.linkage(df[int_col][:1500], method='centroid'))

plt.title('dendrogram')

plt.xlabel('price')

plt.ylabel('availability')

plt.show()