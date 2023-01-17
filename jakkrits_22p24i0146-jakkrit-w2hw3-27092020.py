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
# !pip install folium
import numpy as np
import pandas as pd

import scipy 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial import distance

from pylab import rcParams
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

import folium

np.set_printoptions(precision=4, suppress=True)
plt.figure(figsize=(12,6))
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1, inplace=True)
df.info()
df['reviews_per_month'].isnull().sum()
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df.info()
df.head()
# limit memory usage

df = df.sample(1000, random_state=42)
df.info()
features = pd.get_dummies(df)
features
feature_names = features.columns
feature_values = features.values
print(feature_names)
print(feature_values[:5])
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
model = AgglomerativeClustering(n_clusters=10, linkage='ward')
model.fit_predict(feature_values)

df['cluster_ward10'] = model.labels_
display(df.head())
cluster_sizes = df.groupby('cluster_ward10').size()
cluster_sizes
cluser_means = df.groupby('cluster_ward10').mean()
cluser_means.T
linked = linkage(feature_values, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=model.labels_,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
