# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import ndimage 

from scipy.cluster import hierarchy 

from scipy.spatial import distance_matrix 

from matplotlib import pyplot as plt 

from sklearn import manifold, datasets 

from sklearn.cluster import AgglomerativeClustering 

from sklearn.datasets.samples_generator import make_blobs 

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
plt.scatter(X1[:, 0], X1[:, 1], marker='o') 
X1.shape
y1.shape
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
agglom.fit(X1, y1)
# Create a figure of size 6 inches by 4 inches.

plt.figure(figsize=(10,6))



# These two lines of code are used to scale the data points down,

# Or else the data points will be scattered very far apart.



# Create a minimum and maximum range of X1.

x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)



# Get the average distance for X1.

X1 = (X1 - x_min) / (x_max - x_min)



# This loop displays all of the datapoints.

for i in range(X1.shape[0]):

    # Replace the data points with their respective cluster value 

    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)

    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),

             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),

             fontdict={'weight': 'bold', 'size': 9})

    

# Remove the x ticks, y ticks, x and y axis

#plt.xticks([])

#plt.yticks([])

#plt.axis('off')







# Display the plot of the original data before clustering

plt.scatter(X1[:, 0], X1[:, 1], marker='.')

# Display the plot

plt.show()
dist_matrix = distance_matrix(X1,X1) 

print(dist_matrix)
Z = hierarchy.linkage(dist_matrix, 'complete')
plt.figure(figsize=(10,6))

dendro = hierarchy.dendrogram(Z)
Z = hierarchy.linkage(dist_matrix, 'average')

plt.figure(figsize=(10,6))

dendro = hierarchy.dendrogram(Z)
import pandas as pd

df = pd.read_csv("../input/cars_clus.csv")
print ("Shape of dataset: ", df.shape)



df.head(5)
# Quick Cleaning



print ("Shape of dataset before cleaning: ", df.size)

df[[ 'sales', 'resale', 'type', 'price', 'engine_s',

       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',

       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',

       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',

       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')

df = df.dropna()

df = df.reset_index(drop=True)

print ("Shape of dataset after cleaning: ", df.size)

df.head(5)
# Quick Feature Selection



featureset = df[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
# Now we can normalize the feature set. __MinMaxScaler__ transforms features by scaling each feature to a given range. It is by default (0, 1). 

from sklearn.preprocessing import MinMaxScaler

x = featureset.values #returns a numpy array

min_max_scaler = MinMaxScaler()

feature_mtx = min_max_scaler.fit_transform(x)

feature_mtx [0:5]
from scipy.spatial import distance_matrix 

dist_matrix = distance_matrix(feature_mtx,feature_mtx) 

print(dist_matrix)
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')

agglom.fit(feature_mtx)

agglom.labels_
df['cluster_'] = agglom.labels_

df.head()
import matplotlib.cm as cm

n_clusters = max(agglom.labels_)+1

colors = cm.rainbow(np.linspace(0, 1, n_clusters))

cluster_labels = list(range(0, n_clusters))



# Create a figure of size 6 inches by 4 inches.

plt.figure(figsize=(16,14))



for color, label in zip(colors, cluster_labels):

    subset = df[df.cluster_ == label]

    for i in subset.index:

            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 

    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)

#    plt.scatter(subset.horsepow, subset.mpg)

plt.legend()

plt.title('Clusters')

plt.xlabel('horsepow')

plt.ylabel('mpg')
df.groupby(['cluster_','type'])['cluster_'].count()
agg_cars = df.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()

agg_cars
plt.figure(figsize=(16,10))

for color, label in zip(colors, cluster_labels):

    subset = agg_cars.loc[(label,),]

    for i in subset.index:

        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')

    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))

plt.legend()

plt.title('Clusters')

plt.xlabel('horsepow')

plt.ylabel('mpg')
