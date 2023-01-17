# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')
data.head()
# As you can see there is no labels in data
x = data['pelvic_radius']
y = data['degree_spondylolisthesis']
plt.figure(figsize=(13,5))
plt.scatter(x,y)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
df = data.loc[:, ['degree_spondylolisthesis', 'pelvic_radius']]
df.head()
# which k value to choose
from sklearn.cluster import KMeans
wcss = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_) # kmeans.inertia : calculate wcss
    
plt.plot(range(1,15), wcss, '-o')
plt.xlabel('number of k (cluster) value')
plt.ylabel('wcss')
plt.show()
# for k=2, lets write KMeans
from sklearn.cluster import KMeans
kmeans2 = KMeans(n_clusters = 2)
clusters =kmeans2.fit_predict(df) # fit first and then predict

# add labels for df
df['label'] = clusters
# plot
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()
# if we choose k=3
from sklearn.cluster import KMeans
kmeans3 = KMeans(n_clusters = 3)
clusters =kmeans3.fit_predict(df) # fit first and then predict

# add labels for df
df['label'] = clusters

# plot
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()
# if we choose k=4
from sklearn.cluster import KMeans
kmeans4 = KMeans(n_clusters = 4)
clusters =kmeans4.fit_predict(df) # fit first and then predict

# add labels for df
df['label'] = clusters

# plot
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()
# plot
colors = [0 if i=='Abnormal' else 1 for i in data['class']] # to create colors
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = colors)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
# DENDOGRAM 
# here we will try to predict how many clusters we have 
from scipy.cluster.hierarchy import linkage, dendrogram # linkage: create dendrogram
df1 = data.loc[:, ['pelvic_radius', 'degree_spondylolisthesis']]
merg = linkage(df1, method='ward') # ward: cluster icindeki yayilimlari minimize et (wcss gibi bisey)
dendrogram(merg, leaf_rotation=90)
plt.xlabel('data points')
plt.ylabel('euclidian distance')
plt.show()
from sklearn.cluster import AgglomerativeClustering

hierarcical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster = hierarcical_cluster.fit_predict(df1)

# add label for df1
df1['label'] = cluster

#plot
plt.scatter(df1['pelvic_radius'],df1['degree_spondylolisthesis'],c = cluster)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
