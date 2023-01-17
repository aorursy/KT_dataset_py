import random 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
%matplotlib inline
cust_df = pd.read_csv('../input/customer-segmentation/Cust_Segmentation.csv')
cust_df.head()
df = cust_df.drop('Address', axis=1)
df.head()
X = df.values[:, 1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet
cluster_num = 3
k_means = KMeans(init='k-means++', n_init=12, n_clusters=cluster_num)
k_means.fit(X)
labels = k_means.labels_
print(labels)
df['Clus_km'] = labels
df.head()
df.groupby('Clus_km').mean()
area = np.pi * (X[:, 1])**2
plt.scatter(X[:,0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .91, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 2], X[:, 3], c=labels.astype(np.float))
