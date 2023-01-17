import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

%matplotlib inline
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.info()
df.head()
df.isnull().sum().sum()
x = df.iloc[:, [3, 4]].values
p = []



for i in range(1,11):

    km = KMeans(n_clusters = i)

    km.fit(x)

    p.append(km.inertia_)

    

plt.plot([i for i in range(1,11)],p)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('Performance')

plt.show()
km = KMeans(n_clusters = 5, random_state = 0)

y = km.fit_predict(x)
plt.scatter(x[:, 0],x[:, 1], c = y, cmap = 'Pastel2' )

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 10, c = '#ff4410')