# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
df_target = pd.read_csv('../input/target.csv', encoding='ISO-8859-1')
print(df_target.shape)
df_target.head(10)
list(df_target)
# choose columns
df_target = df_target[['Name', 'Address.Latitude', 'Address.Longitude']]
df_target = df_target.set_index('Name')
df_target.head(10)
coords = df_target.values
# within 100 km
kms_per_radian = 6371.0088
epsilon = 300 / kms_per_radian
epsilon
db = DBSCAN(eps = epsilon,
             min_samples = 3,
             algorithm = 'ball_tree',
             metric = 'haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
cluster_labels
num_clusters
df_target['DBSCAN_labels'] = cluster_labels
df_target.head()
df_target['DBSCAN_labels'].value_counts().sort_index()
plt.scatter(df_target['Address.Latitude'], df_target['Address.Longitude'])
plt.show()
import seaborn as sns; sns.set()
plt.figure(figsize = (15, 10))
ax = sns.scatterplot(x="Address.Latitude", 
                     y="Address.Longitude", 
                     hue="DBSCAN_labels",
                     data=df_target)
plt.xlabel('Target Address Latitude')
plt.ylabel('Target Address Longitude')
ax
kms_per_radian = 6371.0088
epsilon = 200 / kms_per_radian
epsilon

db = DBSCAN(eps = epsilon,
             min_samples = 3,
             algorithm = 'ball_tree',
             metric = 'haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))

df_target['DBSCAN_labels'] = cluster_labels

import seaborn as sns; sns.set()
plt.figure(figsize = (15, 10))
ax = sns.scatterplot(x="Address.Latitude", 
                     y="Address.Longitude", 
                     hue="DBSCAN_labels",
                     data=df_target)
plt.xlabel('Target Address Latitude')
plt.ylabel('Target Address Longitude')
ax
df_target['DBSCAN_labels'].value_counts().sort_index()
epsilon
num_clusters
