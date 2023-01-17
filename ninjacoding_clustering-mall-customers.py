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
# Import the libraries

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn.cluster import KMeans

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN 

from sklearn.preprocessing import StandardScaler 

from sklearn.cluster import Birch

from sklearn.cluster import MeanShift

from sklearn.mixture import GaussianMixture

from sklearn.cluster import AffinityPropagation

from sklearn.cluster import OPTICS
data = pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')
X = data.iloc[:, [3, 4]].values
plt.figure(figsize=(15,10))

sns.set(style = 'whitegrid')

plt.title('Distribution',fontsize=15)

plt.axis('off')

sns.distplot(X,color='orange')

plt.xlabel('Distribution')

plt.ylabel('#Customers')
plt.figure(figsize=(22,10))

sns.countplot(data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Distribution of Annual Income (k$)', fontsize = 20)

plt.show()
plt.figure(figsize=(15,10))

sns.set(style = 'whitegrid')

sns.distplot(data['Annual Income (k$)'],color='g')

plt.title('Distribution of Annual Income(k$)', fontsize = 15)

plt.xlabel('Range of Annual Income(k$)')

plt.ylabel('#Customers')
plt.figure(figsize=(15,10))

sns.countplot(data['Age'], palette = 'cool')

plt.title('Distribution of Age', fontsize = 20)

plt.show()
plt.figure(figsize=(15,10))

sns.set(style = 'whitegrid')

sns.distplot(data['Age'],color='m')

plt.title('Distribution of Age', fontsize = 15)

plt.xlabel('Range of Age')

plt.ylabel('#Customers')
plt.figure(figsize=(25,10))

sns.countplot(data['Spending Score (1-100)'], palette = 'copper')

plt.title('Distribution of Spending Score (1-100)', fontsize = 20)

plt.show()
plt.figure(figsize=(15,10))

sns.set(style = 'whitegrid')

sns.distplot(data['Spending Score (1-100)'],color='r')

plt.title('Distribution of Spending Score (1-100)', fontsize = 15)

plt.xlabel('Range of Spending Score (1-100)')

plt.ylabel('#Customers')
plt.figure(figsize=(10,10))

size = data['Genre'].value_counts()

colors = ['pink', 'yellow']

plt.pie(size, colors = colors, explode = [0, 0.15], labels = ['Female', 'Male'], shadow = True, autopct = '%.2f%%')

plt.title('Gender', fontsize = 15)

plt.axis('off')

plt.legend()

plt.show()
sns.catplot(x="Genre", kind="count", palette="cool", data=data)
plt.figure(figsize=(15,10))

plt.title('Heatmap',fontsize=15)

sns.heatmap(X,cmap='BuPu')
plt.figure(figsize=(15,10))

sns.heatmap(data.corr(), cmap = 'Greens', annot = True)

plt.title('Heatmap for the Data', fontsize = 15)

plt.show()
sns.pairplot(data)
# Use of the elbow method to find the optimal number of clusters

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(15,15))

plt.scatter(range(1, 11),wcss,c='b',s=100)

plt.plot(range(1, 11),wcss,c='r',linewidth=4)

plt.title('The Elbow Method',fontsize=20)

plt.xlabel('Number of clusters',fontsize=20)

plt.ylabel('Within-cluster-sum-of-squares',fontsize=20)

plt.show()
# Train the K-Means model on the dataset

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(X)
# Visualization of the clusters of customers

plt.figure(figsize=(15,15))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 60, c = 'c', label = '#1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 60, c = 'g', label = '#2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 60, c = 'm', label = '#3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 60, c = 'b', label = '#4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 60, c = 'r', label = '#5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')

plt.title('Clusters',fontsize=20)

plt.xlabel('Annual Income (k$)',fontsize=20)

plt.ylabel('Spending Score (1-100)',fontsize=20)

plt.legend()

plt.show()
# Use the dendrogram to find the optimal number of clusters

plt.figure(figsize=(25,15))

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendrogram',fontsize=20)

plt.xlabel('Customers',fontsize=20)

plt.ylabel('Euclidean distances',fontsize=20)

plt.show()
# Train the Hierarchical Clustering model on the dataset

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(X)
# Visualization of the clusters of customers

plt.figure(figsize=(15,15))

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 60, c = 'm', label = '#1')

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 60, c = 'r', label = '#2')

plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 60, c = 'b', label = '#3')

plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 60, c = 'g', label = '#4')

plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 60, c = 'c', label = '#5')

plt.title('Clusters',fontsize=20)

plt.xlabel('Annual Income (k$)',fontsize=20)

plt.ylabel('Spending Score (1-100)',fontsize=20)

plt.legend()

plt.show()
def dbscan(X, eps, min_samples):

    ss = StandardScaler()

    X = ss.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)

    db.fit(X)

    y_pred = db.fit_predict(X)

    plt.figure(figsize=(10,10))

    plt.title('Clusters',fontsize=20)

    plt.xlabel('Annual Income (k$)',fontsize=20)

    plt.ylabel('Spending Score (1-100)',fontsize=20)

    plt.legend()

    plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='coolwarm')

    plt.title("DBSCAN")
dbscan(X,eps=0.275,min_samples=5)
brc = Birch(n_clusters=5)

brc.fit(X)

brc_y_pred = brc.predict(X)

plt.figure(figsize=(10,10))

plt.title('Clusters',fontsize=20)

plt.xlabel('Annual Income (k$)',fontsize=20)

plt.ylabel('Spending Score (1-100)',fontsize=20)

plt.legend()

plt.scatter(X[:,0], X[:,1],c=brc_y_pred, cmap='coolwarm')

plt.title("BIRCH")
gmm = GaussianMixture(n_components=5)

gmm.fit(X)

gmm_y_pred = gmm.predict(X)

plt.figure(figsize=(10,10))

plt.title('Clusters',fontsize=20)

plt.xlabel('Annual Income (k$)',fontsize=20)

plt.ylabel('Spending Score (1-100)',fontsize=20)

plt.legend()

plt.scatter(X[:,0], X[:,1],c=gmm_y_pred, cmap='coolwarm')

plt.title("Gaussian Mixture Model")
ap = AffinityPropagation(random_state=0)

ap.fit(X)

ap_y_pred = ap.predict(X)

plt.figure(figsize=(10,10))

plt.title('Clusters',fontsize=20)

plt.xlabel('Annual Income (k$)',fontsize=20)

plt.ylabel('Spending Score (1-100)',fontsize=20)

plt.legend()

plt.scatter(X[:,0], X[:,1],c=ap_y_pred, cmap='spring')

plt.title("Affinity Propagation Model")
ms = MeanShift(bandwidth=2)

ms.fit(X)

ms_y_pred = ms.predict(X)

plt.figure(figsize=(10,10))

plt.title('Clusters',fontsize=20)

plt.xlabel('Annual Income (k$)',fontsize=20)

plt.ylabel('Spending Score (1-100)',fontsize=20)

plt.legend()

plt.scatter(X[:,0], X[:,1],c=ms_y_pred, cmap='seismic')

plt.title("MeanShift Model")
opt = OPTICS(min_samples=5)

opt_y_pred = opt.fit_predict(X)

plt.figure(figsize=(10,10))

plt.title('Clusters',fontsize=20)

plt.xlabel('Annual Income (k$)',fontsize=20)

plt.ylabel('Spending Score (1-100)',fontsize=20)

plt.legend()

plt.scatter(X[:,0], X[:,1],c=opt_y_pred, cmap='inferno')

plt.title("OPTICS Model")