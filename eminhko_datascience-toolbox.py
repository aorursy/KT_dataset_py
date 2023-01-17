# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#To generate clustering and classification dataset
from sklearn.datasets import make_blobs

n_samples = 10000
centers = [(-2.5, -2.5), (2.5, 2.5), (7.5, -2.5)]

X,y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0, centers=centers, random_state=1)
%matplotlib inline
import matplotlib.pyplot as plt

#Kmeans for clustering
from sklearn.cluster import KMeans

# For different number of clusters
for numclust in [2,4,6,8,10]:
    #Create a model and train it
    kmeans = KMeans(n_clusters=numclust, n_init=10)
    cluster_labels= kmeans.fit_predict(X)
    plt.figure(figsize=(12,8))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
    plt.title("Clustering with %2i clusters" % numclust)
    plt.show()
import numpy as np

kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


h = .02
xx, yy = np.meshgrid(np.arange(-10, 15, h), np.arange(-8, 8, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])


Z = Z.reshape(xx.shape)
plt.figure(figsize=(12,8))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c=kmeans.predict(kmeans.cluster_centers_), s=100)
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
x_new = np.array([[-7,-1],[2,-2],[8,-1],[4,-1]])

plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.scatter(x_new[:,0], x_new[:,1], c=kmeans.predict(x_new), s=500)
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

# Create dataset
sample = 1000
X,y = make_classification(n_samples= sample, n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=2, random_state=0)


#Plot
plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 1], marker='x', c=y)
plt.xlim((-3,3))
plt.ylim((-3,3))
plt.show()

h = .02
xx, yy = np.meshgrid(np.arange(-3,3, h), np.arange(-3,3, h))
#Classification algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


#Decision Tree Plot
clf_dt = DecisionTreeClassifier(min_samples_leaf=20)
clf_dt.fit(X,y)

Z = clf_dt.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(figsize=(12,8))
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
plt.title("Decision Tree")
plt.show()

#Random Forest Plot
clf_rf = RandomForestClassifier(n_estimators = 100)
clf_rf.fit(X,y)

Z = clf_rf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(figsize=(12,8))
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
plt.title("Random Forest")
plt.show()



#Gradient Boosting Plot
clf_gb = GradientBoostingClassifier(n_estimators = 10)
clf_gb.fit(X,y)

Z = clf_gb.predict(np.c_[xx.ravel(), yy.ravel()])


Z = Z.reshape(xx.shape)
plt.figure(figsize=(12,8))

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
plt.title("Gradient Boosting")
plt.show()

#2 new observations
x_new = np.array([[4,-1],[5,2]])

for model in [clf_dt, clf_rf, clf_gb]:
    plt.figure(figsize=(12,8))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(x_new[:,0], x_new[:,1],c=model.predict(x_new), s=500)
    plt.title("Classification with %2s" % model)
    plt.show()