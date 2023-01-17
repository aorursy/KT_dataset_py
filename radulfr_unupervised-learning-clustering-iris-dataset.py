# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv")
df.head()
dfIds = df['Id']
df.drop(["Id"],axis=1,inplace=True)
df.tail()
sns.pairplot(data=df,hue="Species",palette="Set2")
plt.show()
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
#This is a dummy function used just for generate a random color in HEX code. 

import random
r = lambda: random.randint(0,255)
getRandomColor = lambda: '#%02X%02X%02X' % (r(),r(),r())

from sklearn.cluster import KMeans
X = df[features]
k=3
km = KMeans(n_clusters=k, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)

X['y_km'] = km.fit_predict(X)

print(features)
km.cluster_centers_

# SepalLengthCm = 0
# SepalWidthCm = 1
# PetalLengthCm = 2
# PetalWidthCm = 3
# Remember K

plt.clf()
for i in range(k):
    plt.scatter(X[X['y_km']== i].PetalLengthCm,
                X[X['y_km']== i].PetalWidthCm,
                s=50, c=getRandomColor(),
                marker='s', edgecolor='black',
                label='Cluster %d' %i)

plt.scatter(km.cluster_centers_[:, 2],#PetalLengthCm
            km.cluster_centers_[:, 3],#PetalWidthCm
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()
X.head()

from sklearn.cluster import KMeans
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(df[features])
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
k=3
km = KMeans(n_clusters=k, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(df[features])

cluster_labels = np.unique(y_km)

silhouette_vals = silhouette_samples(df[features], y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / k)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

#Methods available: {'complete', 'single', 'average', 'ward'}
#Distances available: {'euclidean', 'cityblock','seuclidean', 'sqeuclidean', 'minkowski', ...}
myMethod='complete'
distance='euclidean'

#More info https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
#distances https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
merg = linkage(df[features],metric=distance, method=myMethod)
plt.figure(figsize=(18,6))
dend = dendrogram(merg, leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidian distance")

plt.suptitle("DENDROGRAM",fontsize=18)
plt.show()
#Example from the Chapter 11 of Python Machine Learning book, from Sebastian Raschka and Vahid Mirjalili

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=getRandomColor())
plt.tight_layout()
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# First, K-means
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)

# And... paint it!
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            edgecolor='black',
            c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            edgecolor='black',
            c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('K-means clustering')

# Finally Agglomerative
ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
y_ac = ac.fit_predict(X)

# And draw it!
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c=getRandomColor(),
            edgecolor='black',
            marker='o', s=40, label='cluster 1')
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c=getRandomColor(),
            edgecolor='black',
            marker='s', s=40, label='cluster 2')
ax2.set_title('Agglomerative clustering')


plt.legend()
plt.tight_layout()
plt.show()
#Dendogram
merg = linkage(X,metric=distance, method=myMethod)
plt.figure(figsize=(18,6))
dend = dendrogram(merg, leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidian distance")

plt.suptitle("DENDROGRAM",fontsize=18)
plt.show()
#---------------------------------
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1],
            c=getRandomColor(), marker='o', s=40,
            edgecolor='black', 
            label='cluster 1')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
            c=getRandomColor(), marker='s', s=40,
            edgecolor='black', 
            label='cluster 2')
plt.legend()
plt.tight_layout()
plt.show()
