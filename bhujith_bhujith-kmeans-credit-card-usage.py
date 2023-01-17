import numpy as np 

import random as rnd

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.datasets.samples_generator import make_blobs

import pylab as pl

%matplotlib inline

import os

print(os.listdir("../input"))

df=pd.read_csv("../input/CreditCardUsage.csv")

df.head(5)
df.info()
df.describe()
df.columns
df.duplicated().sum()
df.isna().sum()
nans=lambda df:df[df.isnull().any(axis=1)]

nans(df)
payment=[0.000000]

df[df.PAYMENTS.isin(payment)].count()
df.MINIMUM_PAYMENTS.fillna(0,inplace=True)
df.isna().sum()
df[df.CREDIT_LIMIT.isnull()]
df.drop(columns="CUST_ID",inplace=True)
df.drop(df.loc[df.CREDIT_LIMIT.isnull()].index,inplace=True)
from sklearn.preprocessing import StandardScaler

new_df=pd.DataFrame(StandardScaler().fit_transform(df),columns=df.columns,index=df.index)
X=new_df

k_means=KMeans(init="k-means++",n_clusters=3,n_init=12)

k_means.fit(X)
labels=k_means.labels_

print(labels)
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

score

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()



Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]

score

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Sum of within sum square')

pl.title('Elbow Curve')

pl.show()
new_df["Clus_km"] = labels

new_df.head(5)
new_df.groupby('Clus_km').mean()
df_7_clusters=new_df
k_means_new=KMeans(init="k-means++",n_clusters=7,n_init=12)

k_means_new.fit(df_7_clusters)
labels=k_means_new.labels_

print(labels)
df_7_clusters["Clus_km"] = labels

df_7_clusters.head(5)
df_7_clusters.groupby('Clus_km').mean()
area=np.pi*(df_7_clusters.iloc[:,14])**3

plt.scatter(df_7_clusters.iloc[:,3],df_7_clusters.iloc[:,13],s=area,c=labels.astype(np.float),alpha=0.5)

plt.xlabel('Purchases',fontsize=18)

plt.ylabel('Payment',fontsize=16)

plt.show()

from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure(1, figsize=(3, 2))

plt.clf()

ax = Axes3D(fig, rect=[10, 10, 0.95,1], elev=48, azim=134)



plt.cla()

ax.set_xlabel('Purchases')

ax.set_ylabel('Credit Limit')

ax.set_zlabel('Payment')



ax.scatter(df_7_clusters.iloc[:, 3],df_7_clusters.iloc[:,12],df_7_clusters.iloc[:, 13], c= labels.astype(np.float))
plt.figure(figsize=(8,5))

plt.title("Purchases vs Payment correlation",fontsize=18)

plt.xlabel ("Purchases",fontsize=14)

plt.ylabel ("Payments",fontsize=14)

plt.grid(True)

plt.scatter(df_7_clusters['PURCHASES'],df_7_clusters['PAYMENTS'],color='red',edgecolor='k',alpha=0.6, s=100)

plt.show()
X_dendogram = df_7_clusters.iloc[:,[3,13]].values

import scipy.cluster.hierarchy as sch



plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('Purchases')

plt.ylabel('Payments')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_dendogram, method = 'ward'))

plt.show()

plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('Purchases')

plt.ylabel('Payments')

plt.hlines(y=190,xmin=0,xmax=2000,lw=3,linestyles='--')

plt.text(x=900,y=220,s='Horizontal line crossing 5 vertical lines',fontsize=20)

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_dendogram, method = 'ward'))

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(X_dendogram)
plt.figure(figsize=(12,7))

plt.scatter(X_dendogram[y_hc == 0, 0], X_dendogram[y_hc == 0, 1], s = 100, c = 'red', label = 'Low purchases average payment')

plt.scatter(X_dendogram[y_hc == 1, 0], X_dendogram[y_hc == 1, 1], s = 100, c = 'blue', label = 'Below average puchase,low payment')

plt.scatter(X_dendogram[y_hc == 2, 0], X_dendogram[y_hc == 2, 1], s = 100, c = 'green', label = 'High Puchases high payment')

plt.scatter(X_dendogram[y_hc == 3, 0], X_dendogram[y_hc == 3, 1], s = 100, c = 'orange', label = 'Average purchase,average payment')

plt.scatter(X_dendogram[y_hc == 4, 0], X_dendogram[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Low purchase,Average payment')

plt.scatter(X_dendogram[y_hc == 5, 0], X_dendogram[y_hc == 5, 1], s = 100, c = 'yellow', label = 'Above Average purchases')

plt.scatter(X_dendogram[y_hc == 6, 0], X_dendogram[y_hc == 6, 1], s = 100, c = 'pink', label = 'Low Purchase,Low payment')

plt.title('Clustering of Purchases vs Payments',fontsize=20)

plt.xlabel('Purchases',fontsize=16)

plt.ylabel('Payments',fontsize=16)

plt.legend(fontsize=16)

plt.grid(True)

plt.axhspan(ymin=20,ymax=25,xmin=0.4,xmax=0.96,alpha=0.3,color='yellow')

plt.show()
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 16):

    kmeans_elbow = KMeans(n_clusters = i, init = 'k-means++')

    kmeans_elbow.fit(X_dendogram)

    wcss.append(kmeans_elbow.inertia_)



with plt.style.context(('fivethirtyeight')):

    plt.figure(figsize=(10,6))

    plt.plot(range(1, 16), wcss)

    plt.title('The Elbow Method with k-means++\n',fontsize=25)

    plt.xlabel('Number of clusters')

    plt.xticks(fontsize=20)

    plt.ylabel('WCSS (within-cluster sums of squares)')

    plt.vlines(x=7,ymin=0,ymax=25,linestyles='--')

    plt.text(x=7,y=25,s='7 clusters seem optimal choice \nfrom the elbow position',

             fontsize=25,fontdict={'family':'Times New Roman'})

    plt.show()