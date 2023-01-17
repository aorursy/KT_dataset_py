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
import pandas as pd

dataset = pd.read_csv("../input/adult_numerical_binned1.csv")
# purity_score function

import numpy as np

from sklearn import metrics



def purity_score(y_true, y_pred):

    # compute contingency matrix (also called confusion matrix)

    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # return purity

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
# evaluate function



def evaluate(y_true, y_clusters):

    cm = np.zeros((np.max(y_true) + 1, np.max(y_clusters) + 1))

    for i in range(y_true.size):

        cm[y_true[i], y_clusters[i]] += 1

    purity = 0.

    for cluster in range(cm.shape[1]):  # clusters are along columns

        purity += np.max(cm[:, cluster])

    return purity / y_true.size
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

import pandas as pd

import numpy as np

from itertools import cycle, islice

import matplotlib.pyplot as plt

from pandas.plotting import parallel_coordinates

%matplotlib inline
X = dataset.iloc[:,:-1].values

X = pd.DataFrame(X)

X
Y = dataset['Income']

Y
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
kmeans = KMeans(n_clusters=2)

kmeans.fit(X)
y_pred = kmeans.predict(X)

print(y_pred)
purity_score(Y, y_pred)
from sklearn.metrics.cluster import adjusted_rand_score

print("ARI =", adjusted_rand_score(Y, y_pred))
from sklearn.cluster import AgglomerativeClustering 

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward')

y_hc=hc.fit_predict(X)

#print(y_hc)
print("ARI =", adjusted_rand_score(Y, y_hc))

purity_score(Y, y_hc)
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# cluster the data into five clusters

dbscan = DBSCAN(eps=0.123, min_samples = 2)

clusters = dbscan.fit_predict(X_scaled)

#print(dbscan)

#print(clusters)
#DBSCAN performance:

print("ARI =", adjusted_rand_score(Y, clusters))

purity_score(Y, clusters)