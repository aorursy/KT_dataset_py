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
#Loading the required modules to read the data

import pandas as pd
cereals = pd.read_csv("../input/Cereals.csv")

cereals.head()
cereals['label'] = cereals['name']+'('+ cereals['shelf'].astype(str) + " - " + round(cereals['rating'],2).astype(str)+')'

cereals.drop(['name','shelf','rating'], axis=1, inplace=True)
#check the head

cereals.head()
cereals.describe()
#select all columns except "label"

cereals_label = cereals["label"]

cereals = cereals[cereals.columns.difference(['label'])]
cereals.isnull().sum()
cereals.isnull().sum().sum()


from sklearn.preprocessing import Imputer

mean_Imputer = Imputer()

Imputed_cereals = pd.DataFrame(mean_Imputer.fit_transform(cereals),columns=cereals.columns)

Imputed_cereals

Imputed_cereals.isnull().sum(axis=0)
from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

standardizer.fit(Imputed_cereals)

std_x = standardizer.transform(Imputed_cereals)

std_cereals = pd.DataFrame(std_x,columns=cereals.columns)

std_cereals.head()
std_cereals.describe()
#loading the ethods

from scipy.cluster.hierarchy import linkage,dendrogram

import matplotlib.pyplot as plt



%matplotlib notebook



#preparing linkage matrix

linkage_matrix = linkage(std_cereals, method = "ward", metric = 'euclidean')



##plotting

dendrogram(linkage_matrix,labels=cereals_label.as_matrix())

plt.tight_layout()

plt.show()
from sklearn.cluster import AgglomerativeClustering



##Instantiating object

agg_clust = AgglomerativeClustering(n_clusters=6, affinity = 'euclidean', linkage = 'ward')



##Training model and return class labels

agg_clusters = agg_clust.fit_predict(std_cereals)



##Label - Cluster

agg_result = pd.DataFrame({'labels': cereals_label, "agg_cluster": agg_clusters})

agg_result.head()
from sklearn.cluster import KMeans

kmeans_object = KMeans(n_clusters = 5, random_state=123)

kmeans_object.fit(std_cereals)

kmeans_clusters= kmeans_object.predict(std_cereals)

kmeans_result = pd.DataFrame({"labels":cereals_label, "kmeans_cluster":kmeans_clusters})

kmeans_result.head()
kmeans_object.cluster_centers_
sse = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=10000).fit(std_cereals)

    std_cereals["cluster"] = kmeans.labels_

#print(data["cluster"])

    sse[k] = kmeans.inertia_ #Intertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()