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
import pandas as pd
dataset = pd.read_csv('/kaggle/input/uci-turkiye-student-evaluation-data-set/turkiye-student-evaluation_generic.csv')
dataset.head()
dataset.shape
dataset.info()
dataset.isnull().any()
dataset.describe()
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(dataset['class'])

#distribution of class more of 3 and 13 class
plt.figure(figsize = (15,10))

sns.countplot(dataset['class'],hue = dataset['nb.repeat'])

#The value of repeat for the majority is 1
plt.figure(figsize = (15,10))

sns.countplot(x='class', hue='difficulty', data=dataset)
sns.countplot(x='difficulty', hue='nb.repeat', data=dataset)
plt.figure(figsize=(20, 20))

sns.boxplot(data=dataset.iloc[:,5:31 ])

#Most of the questions related to the instructor has higher range with less spread
data=dataset.iloc[:,5:]

data
#Before performing PCA it is important to perform standard scaling did not do here as all the data was in same scale

from sklearn.decomposition import PCA 



pca = PCA(n_components = None) 

data = pca.fit_transform(data) 

explained_variance = pca.explained_variance_ratio_
explained_variance
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 10):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 10), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

#The number of cluster by elbow method seems to be 2
kmeans = KMeans(n_clusters = 3, init = 'k-means++')

y_kmeans = kmeans.fit_predict(data)



# Visualising the clusters

plt.scatter(data[y_kmeans == 0, 0], data[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')

plt.scatter(data[y_kmeans == 1, 0], data[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')

plt.scatter(data[y_kmeans == 2, 0], data[y_kmeans == 2, 1], s = 100, c = 'red', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'blue', label = 'Centroids')

plt.title('Clusters of students')

plt.xlabel('PCA 1')

plt.ylabel('PCA 2')

plt.legend()

plt.show()
df_cluster = pd.DataFrame(y_kmeans,columns = ['Cluster'])

dataset = dataset.join(df_cluster)
dataset.Cluster.value_counts().plot(kind = 'bar')
import collections

collections.Counter(y_kmeans)
#Now ploting the dendogram for hirearchial clustering

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(data, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('questions')

plt.ylabel('Euclidean distances')

plt.show()
#From the dendogram we can see the appropriate cluster is 2 as the largest vertical distance in the dendogram

#passes through the lines is 2

# Fitting Hierarchical Clustering to the dataset

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(data)

X = data

# Visualising the clusters

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'red', label = 'Cluster 2')

plt.title('Clusters of STUDENTS')

plt.xlabel('PCA 1')

plt.ylabel('PCA 2')

plt.legend()

plt.show()
dataset = dataset.join(pd.DataFrame(y_hc,columns = ['CLuster2']))