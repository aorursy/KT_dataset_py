# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import chardet

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from scipy.cluster import hierarchy

from scipy.spatial import distance_matrix

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import davies_bouldin_score

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# doing this because of UTF-8 error while reading csv file

with open('/kaggle/input/forest-fires-in-brazil/amazon.csv','rb') as rawdata:

    print(chardet.detect(rawdata.read(10000)))
df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding='ISO-8859-1')
# creating dummies variables

#dum = pd.get_dummies(df[['state','month']],drop_first=True)

#df = pd.concat([df,dum],axis=1)

df.drop(['state','month'],axis=1,inplace=True)
# extracting month from date(given in dataset)

df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].dt.month

df.drop('date',axis=1,inplace=True)
ss = StandardScaler()

scaled = ss.fit_transform(df)
pca = PCA(n_components=2)

pca.fit(scaled)

x_pca = pca.transform(scaled)
plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0],x_pca[:,1],cmap='plasma')
# Clustering using K-means 

km = KMeans(n_clusters=2,random_state=1)

fit = km.fit(x_pca)

pred = km.labels_
# Looking for best k value(no. of clusters)

cl = [1,10]

sse = []

for i in cl:

    km = KMeans(n_clusters=i)

    pred = km.fit_predict(df)

    sse.append(km.inertia_)



plt.xlabel('K')

plt.ylabel('squared error')

plt.plot(cl,sse)    
# hierarchy clustering using agglomerative approach

agglo = AgglomerativeClustering(n_clusters=2,linkage='complete')

agglo.fit(x_pca)
agglo_lab = agglo.labels_
# davies_bouldin_score close to 0 in good

davies_bouldin_score(df,agglo_lab)
# silhouette_score close 1 is good 

silhouette_score(df,agglo_lab)
# In this case hierarchy clustering using agglomerative approach is giving best results