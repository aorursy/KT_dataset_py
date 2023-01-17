# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.cluster import KMeans

from sklearn.cluster import AffinityPropagation

from sklearn.mixture import GaussianMixture

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.preprocessing import LabelEncoder 

from sklearn import metrics
col_names = ["CustomerID","Gender","Age","AnnualIncome","spending_score"]

data = pd.read_csv("../input/mall-customers/Mall_Customers.csv",names=col_names,header=0)

data.head()
### Heatmap of dataset

import seaborn as sns

plt.figure(figsize=(10,8))

sns.heatmap(data.corr(), cmap = 'coolwarm', annot = True)

plt.title('Heatmap for the Data', fontsize = 15)

plt.show()
data = data.iloc[:,[1,2,4]]
# Preprocessing



## Label Encoding

enc = LabelEncoder()

data["Gender"] = enc.fit_transform(data["Gender"])
### Scatter Plot

fig = plt.figure(figsize=(14,8))

ax = plt.axes(projection="3d")



x = data["Age"].values

z = data["Gender"].values

y = data["spending_score"].values



img = ax.scatter(x, y, z, s=50)

ax.set_zticks([0, 1])

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score')

ax.set_zlabel('Gender (F = 1, M = 0)')

plt.show()
data.head()
# Kmeans assuming 4 classes as it gave the best silhouette score

vals = data.values

k = 4

k_means = KMeans(n_clusters = k, init = 'k-means++').fit(vals)
centers = k_means.cluster_centers_

centers
from sklearn.metrics import silhouette_score



score = silhouette_score (data, k_means.labels_)



print("Score = ", score)
# Plotting Centroids for K-Means



fig = plt.figure(figsize=(14,8))

ax = plt.axes(projection="3d")



x = data["Age"].values

z = data["Gender"].values

y = data["spending_score"].values



ax.scatter3D(x, y, z, s=100, c=k_means.labels_, edgecolors='b')

ax.scatter3D(k_means.cluster_centers_[:, 1], k_means.cluster_centers_[:, 2],k_means.cluster_centers_[:, 0], s = 300, color = 'green', marker="P", edgecolors='black')



# fig.colorbar(img1)

ax.set_zticks([0, 1])

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score')

ax.set_zlabel('Gender (F = 1, M = 0)')

plt.show()
## Affinity Propogation 

affinity_prop = AffinityPropagation().fit(vals)

affinity_centers = affinity_prop.cluster_centers_indices_

no_of_clusters = len(affinity_centers)

print(f"Centers = {no_of_clusters}")
# Plotting Centroids for affinity propogation



fig = plt.figure(figsize=(14,8))

ax = plt.axes(projection="3d")



x = data["Age"].values

z = data["Gender"].values

y = data["spending_score"].values



ax.scatter3D(x, y, z, s=100, c=affinity_prop.labels_, edgecolors='b')

ax.scatter3D(affinity_prop.cluster_centers_[:, 1], affinity_prop.cluster_centers_[:, 2],affinity_prop.cluster_centers_[:, 0], s = 300, color = 'black', marker="P", edgecolors='red')



# fig.colorbar(img1)

ax.set_zticks([0, 1])

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score')

ax.set_zlabel('Gender (F = 1, M = 0)')

plt.show()
## Mean-Shift Algorithm

bandwidth = estimate_bandwidth(vals, quantile=0.2, n_samples=20)

mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(vals)

mean_shift_centers = mean_shift.cluster_centers_

num_clusters = len(mean_shift_centers)

print(f"No of clusters = {num_clusters}")
### Plotting the centroids



fig = plt.figure(figsize=(14,8))

ax = plt.axes(projection="3d")



x = data["Age"].values

z = data["Gender"].values

y = data["spending_score"].values



ax.scatter3D(x, y, z, s=100, c=mean_shift.labels_, edgecolors='b')

ax.scatter3D(mean_shift.cluster_centers_[:, 1], mean_shift.cluster_centers_[:, 2],mean_shift.cluster_centers_[:, 0], s = 300, color = 'black', marker="P", edgecolors='red')



# fig.colorbar(img1)

ax.set_zticks([0, 1])

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score')

ax.set_zlabel('Gender (F = 1, M = 0)')

plt.show()