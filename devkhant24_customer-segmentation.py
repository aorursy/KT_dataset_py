# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import davies_bouldin_score

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster import hierarchy

from scipy.spatial import distance_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading csv file 

df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.drop('CustomerID',axis=1,inplace=True)
# creating dummy variable 

dum = pd.get_dummies(df['Gender'],drop_first=True)

df = pd.concat([dum,df],axis=1)

df.drop('Gender',axis=1,inplace=True)
df.head()
# Standardizing data using MinMaxScaler

ss = StandardScaler()

mm = MinMaxScaler()

scaler = mm.fit_transform(df)
# Applying princpal compnent analysis for plot data in 2D graph

pca = PCA(n_components=2)

pca.fit(scaler)
xpca = pca.transform(scaler)
# ploting graph of whole data in 2 dimension

plt.figure(figsize=(8,6))

plt.scatter(xpca[:,0],xpca[:,1])
# Using Kmeans for clustering

km = KMeans(n_clusters=2,random_state=1)

km.fit(xpca)

pred = km.predict(xpca)

lab = km.labels_
# finding least error by changing no. of clusters

# This method is known as Elbow Method

a = range(1,10)

sse = []

for i in a:

    km = KMeans(n_clusters=i)

    km.fit(xpca)

    sse.append(km.inertia_)

plt.plot(a,sse)    
# ploting results of clustering

plt.scatter(xpca[:,0],xpca[:,1],c=pred,cmap='Paired')
# metrics used for measuring accuracy of clustering(value closer to 1 better it is)

silhouette_score(xpca,pred)
# metrics used for measuring accuracy of clustering(value closer to 0 better it is)

davies_bouldin_score(xpca,pred)
# Using Agglomerative approach for clustering

agglo = AgglomerativeClustering(n_clusters=2,linkage='average')

agpred = agglo.fit_predict(xpca)
plt.scatter(xpca[:,0],xpca[:,1],c=agpred,cmap='Paired')
agp = agglo.labels_
silhouette_score(xpca,agp)
davies_bouldin_score(xpca,agp)
#  showing hierarchy of data

dist = distance_matrix(xpca,xpca)

z = hierarchy.linkage(xpca,'average')

dendro = hierarchy.dendrogram(z)
# Using Agglomerative approach for clustering

db = DBSCAN()

dbpred = db.fit_predict(xpca)
plt.scatter(xpca[:,0],xpca[:,1],c=dbpred,cmap='Paired')
dbp = db.labels_
silhouette_score(xpca,dbp)
davies_bouldin_score(xpca,dbp)