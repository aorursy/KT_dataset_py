url="https://raw.githubusercontent.com/Gunnvant/PythonForModellers/master/Data/kc_housingdata.csv"
import pandas as pd

import numpy as np
data=pd.read_csv(url)
data.head(2)
### Properties to buy ####

## What are the different groups of properties that exist ##

## price, bedrooms, bathrooms, sqft_living ##

data_rel=data[['price','bedrooms','bathrooms','sqft_living']]
data_rel.head(2)
data_rel.isnull().sum()
data_rel.shape
#### How to do kmeans clustering ####

## scale the data ##

from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
data_scaled=scaler.fit_transform(data_rel)
data_scaled
import sklearn.cluster as cluster
mod=cluster.KMeans(n_clusters=3)
mod=mod.fit(data_scaled)
mod.labels_ ### cluster groups
data_rel['labels_3']=mod.labels_
data_rel.head(2) ### 
data_rel['labels_3'].value_counts()
data_rel.shape
123/21613
#### How do I decide the number of clusters to look at ####
### How many groups I should be looking at ####

## Real estate developer, who wanted to start making properties in this city,

## Context is very important, create very rigid rules while talking about the context

## Mathematical rule to decide the number of segments,

## Mathematical rules of thumb, that can help you decide the number of clusters
### Choice of cluster is guided by context and business problem
## Mathematical rule of thumb, 
mod.inertia_ ###(WSS1+WSS2+WSS3)
compactness=[]

cls=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
for i in cls:

    mod=cluster.KMeans(n_clusters=i)

    mod=mod.fit(data_scaled)

    compactness.append(mod.inertia_)
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(cls,compactness,"*")
### k = think about the context and then arrive at the value of k

### build a scree plot , range of values you can try (5 to 10)

### What will I analyse when I analysing different clusters?
mod3=cluster.KMeans(n_clusters=3)

mod3=mod3.fit(data_scaled)
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
data_rel.head(2)
get_zprofiles(data=data_rel.drop("labels_3",axis=1),kmeans=mod3)
### Cluster 0: Price is low, bedrooms are also low, washrooms are also low

### Cluster 1: Price high, 

### Cluster 2: Only price is low rest is high
get_profiles(data_rel.drop("labels_3",axis=1),mod3)
### Cluster 0: Price is 32% lower than average, Number of bedrooms are 20% lowe
mod5=cluster.KMeans(n_clusters=5)

mod5=mod5.fit(data_scaled)
get_zprofiles(data=data_rel.drop("labels_3",axis=1),kmeans=mod5)
get_profiles(data=data_rel.drop("labels_3",axis=1),kmeans=mod5)
mod6=cluster.KMeans(n_clusters=6)

mod6=mod6.fit(data_scaled)
get_zprofiles(data=data_rel.drop("labels_3",axis=1),kmeans=mod6)
get_profiles(data=data_rel.drop("labels_3",axis=1),kmeans=mod6)
mod8=cluster.KMeans(n_clusters=8)

mod8=mod8.fit(data_scaled)
get_zprofiles(data=data_rel.drop("labels_3",axis=1),kmeans=mod8)
get_profiles(data=data_rel.drop("labels_3",axis=1),kmeans=mod8)