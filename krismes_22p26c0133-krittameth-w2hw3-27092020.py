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
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
############### Dataset shape ###############
df.shape
df.dtypes
df = df.dropna()
df.shape
nbh_lst1 = df.neighbourhood_group.unique().tolist()
nbh_lst2 = df.neighbourhood.unique().tolist()
room_type = df.room_type.unique().tolist()
nbh1 = {}
nbh2 = {}
room = {}
for i,j in enumerate(nbh_lst1):
    nbh1[j] = i
for i,j in enumerate(nbh_lst2):
    nbh2[j] = i
for i,j in enumerate(room_type):
    room[j] = i
df['neighbourhood_group'].replace(nbh1,inplace=True)
df['neighbourhood'].replace(nbh2,inplace=True)
df['room_type'].replace(room,inplace=True)
df.head()
df.dtypes
df.head()
featureset = df[['neighbourhood_group','neighbourhood','room_type','price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']].sample(2000)
df.size
from sklearn.preprocessing import MinMaxScaler
x = featureset.values 
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]
from scipy.spatial import distance_matrix
dist_matrix = distance_matrix(feature_mtx,feature_mtx) 
print(dist_matrix)
from sklearn import datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 
from scipy.cluster.hierarchy import dendrogram
agglom = AgglomerativeClustering(n_clusters = None, distance_threshold=0)
agglom = agglom.fit(feature_mtx)
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
from matplotlib import pyplot as plt
plt.figure(figsize=(20,14))
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(agglom, truncate_mode='level', p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
