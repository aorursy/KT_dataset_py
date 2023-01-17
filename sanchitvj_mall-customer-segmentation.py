import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler 

from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.mixture import GaussianMixture

from sklearn.cluster import AffinityPropagation

from mpl_toolkits.mplot3d import Axes3D
K = 15

col_names = ["customer_id","gender","age","annual_income","spending_score"]

data = pd.read_csv("../input/mall-customers/Mall_Customers.csv",names=col_names,header=0)

data = data.sample(frac=1)

data.head()
X = data.iloc[:, [3, 4]].values
plt.figure(figsize=(12,8))

sns.set(style = 'darkgrid')

plt.title('Distribution',fontsize=15)

plt.axis('off')

sns.distplot(X,color='cyan')

plt.xlabel('Distribution')

plt.ylabel('#Customers')
plt.figure(figsize=(20,8))

sns.countplot(data['annual_income'], palette = 'rainbow')

plt.title('Distribution of Annual Income (k$)', fontsize = 20)

plt.show()
plt.figure(figsize=(8,8))

size = data['gender'].value_counts()

colors = ['magenta', 'blue']

plt.pie(size, colors = colors, explode = [0, 0.15], labels = ['Female', 'Male'], shadow = True, autopct = '%.2f%%')

plt.title('Gender', fontsize = 15)

plt.axis('off')

plt.legend()

plt.show()
plt.figure(figsize=(10,8))

sns.heatmap(data.corr(), cmap = 'coolwarm', annot = True)

plt.title('Heatmap for the Data', fontsize = 15)

plt.show()
data = data.iloc[:,[1,2,4]]

data = data.replace(to_replace="Female",value=1)

data = data.replace(to_replace="Male",value=0)
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
# Use of the elbow method to find the optimal number of clusters

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,8))

plt.scatter(range(1, 11),wcss,c='b',s=100)

plt.plot(range(1, 11),wcss,c='r',linewidth=4)

plt.title('The Elbow Method',fontsize=20)

plt.xlabel('Number of clusters',fontsize=20)

plt.ylabel('Within-cluster-sum-of-squares',fontsize=20)

plt.show()
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