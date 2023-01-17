# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

x,y=make_blobs(n_samples=500,n_features=2,centers=4,random_state=0)
fig,ax1=plt.subplots(1)

ax1.scatter(x[:,0],x[:,1]

           ,marker='o'

           ,s=8)

plt.show()
color=['red','pink','orange','gray']

fig,ax1=plt.subplots(1)

for i in range(4):

    ax1.scatter(x[y==i,0],x[y==i,1]

               ,marker='o'

               ,s=8

               ,c=color[i]

               )

plt.show()    
from sklearn.cluster import KMeans

n_clusters=3
cluster=KMeans(n_clusters=n_clusters,random_state=0).fit(x)

y_pred=cluster.labels_
centroid=cluster.cluster_centers_

centroid
color=['red','pink','orange','gray']

fig,ax1=plt.subplots(1)

for i in range(n_clusters):

    ax1.scatter(x[y_pred==i,0],x[y_pred==i,1]

               ,marker='o'

               ,s=8

               ,c=color[i]

               )

ax1.scatter(centroid[:,0],centroid[:,1]

           ,marker='x'

           ,s=15

           ,c='black')

plt.show() 
from sklearn.metrics import silhouette_score

from sklearn.metrics import silhouette_samples
silhouette_score(x,y_pred)
cluster=KMeans(n_clusters=4,random_state=0).fit(x)

silhouette_score(x,cluster.labels_)
cluster=KMeans(n_clusters=5,random_state=0).fit(x)

silhouette_score(x,cluster.labels_)
from sklearn.metrics import calinski_harabaz_score
calinski_harabaz_score(x,y_pred)
n_clusters=4

cluster=KMeans(n_clusters=4,random_state=0).fit(x)

calinski_harabaz_score(x,cluster.labels_)
cluster=KMeans(n_clusters=5,random_state=0).fit(x)

calinski_harabaz_score(x,cluster.labels_)
color=['red','pink','orange','gray']

fig,ax1=plt.subplots(1)

for i in range(n_clusters):

    ax1.scatter(x[cluster.labels_==i,0],x[cluster.labels_==i,1]

               ,marker='o'

               ,s=8

               ,c=color[i]

               )

ax1.scatter(cluster.cluster_centers_[:,0],cluster.cluster_centers_[:,1]

           ,marker='x'

           ,s=15

           ,c='black')

plt.show() 
import matplotlib.cm as cm
for j in np.arange(2,10,2):

    fig,(ax1,ax2)=plt.subplots(1,2)

    fig.set_size_inches(18,7)

    ax1.set_xlim([-0.1,1])

    ax1.set_ylim([0,x.shape[0]+(n_clusters+1)*10])

    n_clusters=j

    cluster=KMeans(n_clusters=n_clusters,random_state=0).fit(x)

    y_pred=cluster.labels_

    silhouette_avg=silhouette_score(x,y_pred)

    print('n_clusters',n_clusters,

         'silhouetteavg',silhouette_avg)

    silhouette_samples_values=silhouette_samples(x,y_pred)

    y_lower=10

    for i in range(n_clusters):

        #when y_pred=i, I could know it's silhouette_samples_values

        ith_silhouette_samples_values=silhouette_samples_values[y_pred==i]

        ith_silhouette_samples_values.sort()

        # the number of items in the cluster i

        size_cluster_i=ith_silhouette_samples_values.shape[0]

        y_upper=y_lower+size_cluster_i

        # set color

        color=cm.nipy_spectral(float(i)/n_clusters)

        # fill the same color bewteen the yaxis range

        ax1.fill_betweenx(np.arange(y_lower,y_upper)

                         ,ith_silhouette_samples_values

                         ,facecolor=color

                         ,alpha=0.7)

        ax1.text(-0.05

                ,y_lower+0.5*size_cluster_i

                ,str(i))

        y_lower=y_upper+10



    ax1.set_title('The Silhouette plot for the various clusters')    

    ax1.set_xlabel('The Silhouette coefficient values')

    ax1.set_ylabel('Cluster labels')



    ax1.axvline(x=silhouette_avg,color='red',linestyle='--')



    ax1.set_yticks([])

    ax1.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])

    # set all points color to 4 types

    color=cm.nipy_spectral(y_pred.astype(float)/n_clusters)



    ax2.scatter(x[:,0],x[:,1]

                ,marker='o'

                ,s=8

                ,c=color

                   )

    ax2.scatter(cluster.cluster_centers_[:,0],cluster.cluster_centers_[:,1]

               ,marker='x'

               ,s=15

               ,c='black')