import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv(
    filepath_or_buffer='../input/rose species.csv',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) 

df.tail()
# Getting the values and plotting it
f1 = df['sepal_len'].values
f2 = df['sepal_wid'].values
f3 = df['petal_len'].values
f4 = df['petal_wid'].values
X = np.array(list(zip(f1, f2, f3, f4)))
plt.scatter(f1, f2, c='black', s=17)
plt.scatter(f3, f4, c='blue', s=17)


from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=2)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

print("Centroid values")
print(centroids) 
k= 2 #no. of clusters
colors = ['r', 'g', 'b', 'y']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], c=colors[i], s=7)
        
ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
