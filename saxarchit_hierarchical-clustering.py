import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



dataset = pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')

dataset.head()
X = dataset.iloc[: , [3, 4]].values
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title('Dendogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')

y_hc = hc.fit_predict(X)
plt.scatter(X[y_hc ==  0, 0], X[y_hc ==  0, 1], s = 100, c = 'red', label = 'cluster1')

plt.scatter(X[y_hc ==  1, 0], X[y_hc ==  1, 1], s = 100, c = 'blue', label = 'cluster2')

plt.scatter(X[y_hc ==  2, 0], X[y_hc ==  2, 1], s = 100, c = 'green', label = 'cluster3')

plt.scatter(X[y_hc ==  3, 0], X[y_hc ==  3, 1], s = 100, c = 'cyan', label = 'cluster4')

plt.scatter(X[y_hc ==  4, 0], X[y_hc ==  4, 1], s = 100, c = 'magenta', label = 'cluster5')

plt.title('Clusters of Clients')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
plt.scatter(X[y_hc ==  0, 0], X[y_hc ==  0, 1], s = 100, c = 'red', label = 'Careful')

plt.scatter(X[y_hc ==  1, 0], X[y_hc ==  1, 1], s = 100, c = 'blue', label = 'Standard')

plt.scatter(X[y_hc ==  2, 0], X[y_hc ==  2, 1], s = 100, c = 'green', label = 'Target')

plt.scatter(X[y_hc ==  3, 0], X[y_hc ==  3, 1], s = 100, c = 'cyan', label = 'Careless')

plt.scatter(X[y_hc ==  4, 0], X[y_hc ==  4, 1], s = 100, c = 'magenta', label = 'Sensible')

plt.title('Clusters of Clients')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()