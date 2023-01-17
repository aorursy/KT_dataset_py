import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv(r'../input/Customer_Segmentation.csv')
df.head()
print(df.isna().sum())
plt.scatter(df.Age,df.Income,c='blue')

plt.xlabel('Age')

plt.ylabel('Income')

plt.show
from sklearn.cluster import KMeans

from sklearn import metrics

from scipy.spatial.distance import cdist

edf=df[['Age',"Income"]].iloc[:,:].values

distortion=[]

intria=[]

kmax=11



for k in range(2,kmax+1):

    kmeans=KMeans(n_clusters=k).fit(edf)

    distortion.append(sum(np.min(cdist(edf,kmeans.cluster_centers_,'euclidean'),axis=1))/edf.shape[0])

    intria.append(kmeans.inertia_)
plt.plot(np.arange(1,11),distortion,'-o')

plt.xlabel('K')

plt.ylabel('Distortion')

plt.show
plt.plot(np.arange(1,11),intria,'-o')

plt.xlabel('K')

plt.ylabel('Intertia')

plt.show
from sklearn.cluster import KMeans

from sklearn import metrics

from sklearn.metrics import silhouette_score

from scipy.spatial.distance import cdist

edf=df[['Age',"Income"]].iloc[:,:].values

sil=[]

kmax=11



for k in range(2,kmax+1):

    kmeans=KMeans(n_clusters=k).fit(edf)

    sil.append(silhouette_score(edf,kmeans.labels_,'euclidean'))
plt.plot(np.arange(1,11),sil,'-o')

plt.xlabel('K')

plt.ylabel('Sil_score')

plt.show
kmeans=KMeans(n_clusters=3).fit(edf)

plt.scatter('Age',"Income",data=df,c=kmeans.labels_)

centroid=kmeans.cluster_centers_

plt.plot(centroid[:,0],centroid[:,1],'o',c='red')

plt.show