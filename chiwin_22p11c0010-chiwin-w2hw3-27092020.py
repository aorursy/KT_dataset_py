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
# Library import 
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler as Sta
# Plot dendrogram function
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
# Read .csv file  
data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.head()
# Select column
df=data[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','price','minimum_nights','number_of_reviews','reviews_per_month']]
df.head()
# Check NaN
df.isna().sum()
# Fill NaN
df = df.fillna(0)
df.isna().sum()
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
new_data_sample = df.sample(n=2500)

sta=Sta()
sta.fit(new_data_sample)
new_data=sta.transform(new_data_sample)
new_data
# Build model
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(new_data)
plt.title('Hierarchical Clustering Dendrogram')
# Plot the top five levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()