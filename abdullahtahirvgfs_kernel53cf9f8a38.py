import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
%matplotlib inline
df = pd.read_csv('../input/admission-predictionkm/admission_predict_update(KM).csv',header=None)
plt.scatter(df[0], df[1], c='olivedrab', label = 'Graph of GRE Score and Research')
df.head()
df.info()
df.columns
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random', random_state = 42)
    kmeans.fit(df.iloc[:, 0:-1]) # Using columns from index 0 to -1 aka Xone and Xtwo
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.show()
kmeans = KMeans(n_clusters=10, n_init=5)
y_kmeans = kmeans.fit_predict(df)
df = np.array(df)
fig = plt.figure(figsize=(15, 15))
ax = fig.gca()

ax.scatter(df[y_kmeans == 0, 0], df[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
ax.scatter(df[y_kmeans == 1, 0], df[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

ax.legend()