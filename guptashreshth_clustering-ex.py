import pandas as pd

import numpy as np

import os

import sklearn.preprocessing as preprocessing

import sklearn.cluster as cluster

import matplotlib.pyplot as plt

%matplotlib inline
os.chdir('../input/')
data=pd.read_csv("kc_housingdata_in.csv")
data.head()
data.shape
## Choose columns that are numeric and have a numeric interpretation

data_num=data[['price','bedrooms','bathrooms','sqft_living']]
data_num.dtypes
## Scale the data using sklearn

dat_scaled=preprocessing.scale(data_num,axis=0)
print (dat_scaled)

print ("Type of output is "+str(type(dat_scaled)))

print ("Shape of the object is "+str(dat_scaled.shape))
kmeans=cluster.KMeans(n_clusters=3,init="k-means++")

kmeans=kmeans.fit(dat_scaled)
kmeans.labels_
kmeans.cluster_centers_
## Elbow method

from scipy.spatial.distance import cdist

K=range(1,20)

wss = []

for k in K:

    kmeans = cluster.KMeans(n_clusters=k,init="k-means++")

    kmeans.fit(dat_scaled)

    wss.append(sum(np.min(cdist(dat_scaled, kmeans.cluster_centers_, 'euclidean'), 

                                      axis=1)) / dat_scaled.shape[0])

plt.plot(K, wss, 'bx')

plt.xlabel('k')

plt.ylabel('Average distortion')

plt.title('Selecting k with the Elbow Method')

plt.show()
import sklearn.metrics as metrics

labels=cluster.KMeans(n_clusters=9,random_state=200).fit(dat_scaled).labels_
metrics.silhouette_score(dat_scaled,labels,metric="euclidean",sample_size=10000,random_state=200)
for i in range(7,13):

    labels=cluster.KMeans(n_clusters=i,random_state=200).fit(dat_scaled).labels_

    print ("Silhoutte score for k= "+str(i)+" is "+str(metrics.silhouette_score(dat_scaled,labels,metric="euclidean",

                                 sample_size=1000,random_state=200)))
data['Cluster_no']=kmeans.labels_
data.head()
def get_zprofiles(data,kmeans):

    data['Labels']=kmeans.labels_

    profile=data.groupby('Labels').mean().subtract(data.drop('Labels',axis=1).mean(),axis=1)

    profile=profile.divide(data.drop('Labels',axis=1).std(),axis=1)

    profile['Size']=data['Labels'].value_counts()

    return profile



def get_profiles(data,kmeans):

    data['Labels']=kmeans.labels_

    profile=data.groupby('Labels').mean().divide(data.drop('Labels',axis=1).mean(),axis=1)

    profile['Size']=data['Labels'].value_counts()

    return profile
## Let's look for profiles for 8,9,10 clusters

kmeans=cluster.KMeans(n_clusters=9,random_state=200).fit(dat_scaled)
get_zprofiles(data=data_num.copy(),kmeans=kmeans)