# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#%% DATA IMPORT

dataset= pd.read_csv("../input/tiny_eeg_self_experiment_music.csv")
#%% Slicing sample data (39647 feature huge )

dataset=dataset.iloc[0:250,:]

dataset.head()
#%% EDA



dataset.info()

dataset.columns
#%% CLEANING DATA

#Drop unnecessary columns

dataset=dataset.drop(['IndexId','Ref1','Ref2', 'Ref3', 'TS1', 'TS2'],axis=1)
dataset.tail()
#%% see on pairplot our dataset every sigle feature pair with the other and non colored like featureless

sns.pairplot(data=dataset)

plt.show()
#%% KMEANS wiht sklearn 

from sklearn.cluster import KMeans

wcss=[]

#find best k value

for k in range(1,15):

    kmeans=KMeans(n_clusters=k)

    kmeans.fit(dataset)

    wcss.append(kmeans.inertia_)

#elbow rule on plot    

plt.figure(figsize=(12,8))

plt.plot(range(1,15),wcss,"-o")

plt.title("wcss / number of cluster", fontsize=18)

plt.xlabel("number of k(cluster) values")

plt.xticks(range(1,15))

plt.grid(True)

plt.ylabel("wcss")

plt.tight_layout()

plt.show()
#%%   from elbow plot we can choose 3 or 4 i'll go with 4 cluster.

#kmeans2=KMeans(n_clusters=2)

#kmeans3=KMeans(n_clusters=3)

kmeans=KMeans(n_clusters=4)

clusters=kmeans.fit_predict(dataset)

dataset["label"]=clusters
#%% plot with cluster / center(centroid)

plt.figure(figsize=(20,8))

plt.scatter(dataset.Channel1[dataset.label==0],dataset.Channel2[dataset.label==0],color="red",alpha= 0.8)

plt.scatter(dataset.Channel1[dataset.label==1],dataset.Channel2[dataset.label==1],color="green",alpha= 0.8)

plt.scatter(dataset.Channel1[dataset.label==2],dataset.Channel2[dataset.label==2],color="blue",alpha= 0.8)

plt.scatter(dataset.Channel1[dataset.label==3],dataset.Channel2[dataset.label==3],color="black",alpha= 0.8)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="orange")# cluster center(centroid)

plt.title("Kmeans(k=4) Cluster")

plt.xlabel("Channel1")

plt.ylabel("Channel2")

plt.legend(dataset,loc='upper right')

plt.show()
#%%   HIERARCICAL CLUSTRING

 # DENDROGRAM

dataset2=dataset

from scipy.cluster.hierarchy import linkage, dendrogram

merg=linkage(dataset2,method="ward")      

dendrogram(merg,leaf_rotation=90)

plt.xlabel("data points")

plt.ylabel("euclidean distanece")

plt.show()
#%%  HIERARCICAL CLUSTRING

# CLUSTRING AND PLOT For n_clusters=4

from sklearn.cluster import AgglomerativeClustering

# AgglomerativeClustering    en alakalı data pointleri cluster ederek tüme varıp yapıp datanın tamamını mantıklı cluster eden algoritma

hierarcical_cluster = AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward")

cluster=hierarcical_cluster.fit_predict(dataset)

dataset2["label2"]=cluster



plt.figure(figsize=(20,8))

plt.xlabel("Channel1")

plt.ylabel("Channel2")

plt.scatter(dataset2.Channel1[dataset2.label2==0],dataset2.Channel2[dataset2.label2==0],color="red",alpha= 0.8)

plt.scatter(dataset2.Channel1[dataset2.label2==1],dataset2.Channel2[dataset2.label2==1],color="green",alpha= 0.8)

plt.scatter(dataset2.Channel1[dataset2.label2==2],dataset2.Channel2[dataset2.label2==2],color="blue",alpha= 0.8)

plt.scatter(dataset2.Channel1[dataset2.label2==3],dataset2.Channel2[dataset2.label2==3],color="black",alpha= 0.8)

#plt.scatter(dataset.Channel1[dataset.label==4],dataset.Channel2[dataset.label==4],color="black",alpha= 0.3)

plt.show()
