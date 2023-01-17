import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

dataset.head(10)
dataset.info()
dataset.describe()
plt.figure(1, figsize = (10, 3))

sns.countplot(y = "Gender", data = dataset)

plt.show()
plt.figure(1, figsize = (15, 5))

sns.barplot(x = "Age", y = "Annual Income (k$)", data = dataset)

plt.show()
plt.figure(1, figsize = (15, 5))

sns.barplot(x = "Age", y = "Spending Score (1-100)", data = dataset)

plt.show()
plt.figure(1, figsize = (10, 5))

sns.distplot(dataset.Age)

plt.show()
sns.pairplot(dataset.drop('CustomerID', axis=1), hue='Gender', aspect=1.5)

plt.show()
from sklearn.cluster import KMeans



clusters = []

X = dataset.drop(['CustomerID', 'Gender'], axis=1)



for i in range(1, 11):

    km = KMeans(n_clusters = i).fit(X)

    clusters.append(km.inertia_)

    

    

fig, ax = plt.subplots(figsize=(12, 8))

sns.lineplot(x = list(range(1, 11)), y = clusters, ax = ax)

ax.set_title('Searching for Elbow')

ax.set_xlabel('Clusters')

ax.set_ylabel('Inertia')





plt.show()

    
km3 = KMeans(n_clusters = 3).fit(X)



X['labels'] = km3.labels_

plt.figure(figsize = (12, 8))



sns.scatterplot(X["Annual Income (k$)"], X["Spending Score (1-100)"], hue = X["labels"], palette=sns.color_palette('hls', 3))

plt.show()
km3 = KMeans(n_clusters = 5).fit(X)



X['labels'] = km3.labels_

plt.figure(figsize = (12, 8))



sns.scatterplot(X["Annual Income (k$)"], X["Spending Score (1-100)"], hue = X["labels"], palette=sns.color_palette('hls', 5))

plt.show()
from sklearn.cluster import AgglomerativeClustering 



agglom = AgglomerativeClustering(n_clusters=5, linkage='average').fit(X)



X['Labels'] = agglom.labels_

plt.figure(figsize = (12, 8))



sns.scatterplot(X["Annual Income (k$)"], X["Spending Score (1-100)"], hue = X["labels"], palette=sns.color_palette('hls', 5))

plt.show()
from sklearn.cluster import DBSCAN 



db = DBSCAN(eps = 11, min_samples = 6).fit(X)



X['Labels'] = db.labels_

plt.figure(figsize = (12, 8))

sns.scatterplot(X['Annual Income (k$)'], X['Spending Score (1-100)'], hue = X['Labels'],palette = sns.color_palette('hls', np.unique(db.labels_).shape[0]))

plt.title('DBSCAN with epsilon 11, min samples 6')

plt.show()
from sklearn.cluster import MeanShift, estimate_bandwidth



# The following bandwidth can be automatically detected using

bandwidth = estimate_bandwidth(X, quantile=0.1)

ms = MeanShift(bandwidth).fit(X)



X['Labels'] = ms.labels_

plt.figure(figsize = (12, 8))

sns.scatterplot(X['Annual Income (k$)'], X['Spending Score (1-100)'], hue = X['Labels'], palette = sns.color_palette('hls', np.unique(ms.labels_).shape[0]))

plt.plot()

plt.title('MeanShift')

plt.show()