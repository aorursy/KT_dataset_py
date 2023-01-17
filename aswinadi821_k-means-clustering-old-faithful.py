import sklearn

from sklearn import cluster

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt 
%matplotlib inline
df=pd.read_csv("../input/faithful.csv")

df.head()
df.columns=['eruptions','waiting']

plt.scatter(df.eruptions,df.waiting)

plt.title('Old Faithful Scatterplot')

plt.xlabel('Length of Eruption(minutes)')

plt.ylabel('Time of Eruption(minutes)')

faith=np.array(df)

k=2

kmeans=cluster.KMeans(n_clusters=k)

kmeans.fit(faith)

labels = kmeans.labels_

centroids = kmeans.cluster_centers_

for i in range(k):

    plt.title('Old Faithful Scatterplot')

    plt.xlabel('Length of Eruption(minutes)')

    plt.ylabel('Time of Eruption(minutes)')



    ds=faith[np.where(labels==i)]

    plt.plot(ds[:,0],ds[:,1],'o',markersize=9)

    lines=plt.plot(centroids[i,0],centroids[i,1],'kx')

    plt.setp(lines,ms=15.0)

    plt.setp(lines,mew=4.0)

plt.show()