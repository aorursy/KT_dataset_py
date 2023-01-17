import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth
from mpl_toolkits.mplot3d import Axes3D

K = 15
col_names = ["customer_id","gender","age","annual_income","spending_score"]
data = pd.read_csv("../input/mall-customers/Mall_Customers.csv",names=col_names,header=0)
data = data.sample(frac=1)
data.head()
data.shape
data = data.iloc[:,[1,2,4]]
data = data.replace(to_replace="Female",value=1)
data = data.replace(to_replace="Male",value=0)
data.head()



fig = plt.figure(figsize=(14,8))
ax = plt.axes(projection="3d")

x = data["age"].values
z = data["gender"].values
y = data["spending_score"].values

img = ax.scatter(x, y, z, s=50)
ax.set_zticks([0, 1])
ax.set_xlabel('Age')
ax.set_ylabel('Spending Score')
ax.set_zlabel('Gender (F = 1, M = 0)')
plt.show()

inertia = []

for k in range(1,K):
  kmeans = KMeans(n_clusters=k,init = 'k-means++')
  kmeans.fit(data.values)
  inertia.append(kmeans.inertia_)

plt.plot(range(1,K), inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Distance of points with centroid')
plt.show()
print("From elbow plot we can find that optimal number of cluster will be 4")
vals = data.values
k = 4 #from graph above
kmeans = KMeans(n_clusters = k, init = 'k-means++').fit(vals)

fig = plt.figure(figsize=(14,8))
ax = plt.axes(projection="3d")

x = data["age"].values
z = data["gender"].values
y = data["spending_score"].values

print(f"K-means: num of clusters - {k}")

ax.scatter3D(x, y, z, s=100, c=kmeans.labels_, edgecolors='b')
ax.scatter3D(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],kmeans.cluster_centers_[:, 0], s = 300, color = 'black', marker="P", edgecolors='red')

# fig.colorbar(img1)
ax.set_zticks([0, 1])
ax.set_xlabel('Age')
ax.set_ylabel('Spending Score')
ax.set_zlabel('Gender (F = 1, M = 0)')
plt.show()
afprop = AffinityPropagation().fit(vals)
cluster_centers = afprop.cluster_centers_indices_
num_clusters = len(cluster_centers)
print(f"Affinity Propagation: num of clusters - {num_clusters}")

fig = plt.figure(figsize=(14,8))
ax = plt.axes(projection="3d")

x = data["age"].values
z = data["gender"].values
y = data["spending_score"].values

ax.scatter3D(x, y, z, s=100, c=afprop.labels_, edgecolors='b')
ax.scatter3D(afprop.cluster_centers_[:, 1], afprop.cluster_centers_[:, 2],afprop.cluster_centers_[:, 0], s = 300, color = 'black', marker="P", edgecolors='red')

# fig.colorbar(img1)
ax.set_zticks([0, 1])
ax.set_xlabel('Age')
ax.set_ylabel('Spending Score')
ax.set_zlabel('Gender (F = 1, M = 0)')
plt.show()
bandwidth = estimate_bandwidth(vals, quantile=0.2, n_samples=20)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(vals)
cluster_centers = ms.cluster_centers_
num_clusters = len(cluster_centers)

print(f"Mean-shift: num of clusters - {num_clusters}")

fig = plt.figure(figsize=(14,8))
ax = plt.axes(projection="3d")

x = data["age"].values
z = data["gender"].values
y = data["spending_score"].values

ax.scatter3D(x, y, z, s=100, c=ms.labels_, edgecolors='b')
ax.scatter3D(ms.cluster_centers_[:, 1], ms.cluster_centers_[:, 2],ms.cluster_centers_[:, 0], s = 300, color = 'black', marker="P", edgecolors='red')

# fig.colorbar(img1)
ax.set_zticks([0, 1])
ax.set_xlabel('Age')
ax.set_ylabel('Spending Score')
ax.set_zlabel('Gender (F = 1, M = 0)')
plt.show()