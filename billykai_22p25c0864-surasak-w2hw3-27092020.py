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
#import module
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler as Sta
import math

%matplotlib inline
#load data set
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.sample(frac=1)
#df=data.copy()
#select data 
df=data[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','price','minimum_nights','number_of_reviews','reviews_per_month']]
df.shape
df.head()
df = df.fillna(0)
df
# Convert data to indexing number
neighbourhood_group_list = pd.unique(df.neighbourhood_group).tolist()
neighbourhood_list = pd.unique(df.neighbourhood).tolist()
room_type_list = pd.unique(df.room_type).tolist()


df['neighbourhood_group'] = pd.Categorical(df['neighbourhood_group'],categories=neighbourhood_group_list)
df['neighbourhood_group'] = df['neighbourhood_group'].cat.codes

df['neighbourhood'] = pd.Categorical(df['neighbourhood'],categories=neighbourhood_list)
df['neighbourhood'] = df['neighbourhood'].cat.codes


df['room_type'] = pd.Categorical(df['room_type'],categories=room_type_list)
df['room_type'] = df['room_type'].cat.codes
df
sample_data=df.sample(n=2500)
sample_data
sta=Sta()
sta.fit(sample_data)
newSample_data=sta.transform(sample_data)
newSample_data
#build hierarchical clustering  model
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model
#fit model to data
model = model.fit(newSample_data)
#check number of clusters
model.n_clusters_
distances = model.distances_
distances.min()
distances.max()
def plot_dendrogram(model, **kwargs):
   
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
#plot dendrogram
plt.title('Hierarchical Clustering Dendrogram')
# plot the top five levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node")
plt.show()
