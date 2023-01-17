import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import MeanShift,estimate_bandwidth,KMeans,AffinityPropagation

%matplotlib inline
data = pd.read_csv('../input/mall-customers/Mall_Customers.csv')

n = LabelEncoder()

data['Genre'] = n.fit_transform(data['Genre'])



data=data.drop(['Annual Income (k$)'],axis=1)

data = data.drop(['CustomerID'],axis=1)



data.head()
bandwidth_ = estimate_bandwidth(data, quantile=0.1, n_samples=len(data))



ms = MeanShift(bandwidth=bandwidth_)

ms.fit(data)
clusters = ms.labels_

cc = ms.cluster_centers_

print(np.unique(clusters))

print(cc)
plt.figure(figsize=(15,10))

ax = plt.axes(projection="3d")

ax.scatter3D(data['Age'],data['Spending Score (1-100)'],data['Genre'],c=clusters.astype(float),s=50)

ax.scatter3D(cc[:,1],cc[:,2],cc[:,0],s=200,color='black',c=None,marker="*")

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score (1-100)')

ax.set_zlabel('Gender')

ax.set_zticks([0,0.5,1])

plt.show()
plt.figure(figsize=(15,10))

ax = plt.axes(projection="3d")

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score (1-100)')

ax.set_zlabel('Gender')



number_of_clusters = len(np.unique(clusters))

c_val = clusters.astype(float)

marker = ".so+xd^1"

for i,marker_shape,c_v in zip(range(number_of_clusters),marker,c_val):

  d = data[clusters==i]

  ax.set_zticks([0,0.5,1])



  ax.scatter3D(d['Age'],d['Spending Score (1-100)'],d['Genre'],marker=marker_shape,s=50)

  cc_v = cc[i]

  ax.scatter3D(cc[:,1],cc[:,2],cc[:,0],s=200,color='black',c=None,marker="*")

plt.show()

model_aff = AffinityPropagation()

model_aff.fit(data)

aff_clusters = model_aff.labels_

aff_cc = model_aff.cluster_centers_

print(aff_cc)

print(np.unique(aff_clusters))



plt.figure(figsize=(15,10))

ax = plt.axes(projection="3d")

ax.scatter3D(data['Age'],data['Spending Score (1-100)'],data['Genre'],c=aff_clusters.astype(float),s=50)

ax.scatter3D(aff_cc[:,1],aff_cc[:,2],aff_cc[:,0],s=200,color='black',c=None,marker="*")

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score (1-100)')

ax.set_zlabel('Gender')

ax.set_zticks([0,0.5,1])

plt.show()

plt.figure(figsize=(15,10))

ax = plt.axes(projection="3d")

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score (1-100)')

ax.set_zlabel('Gender')



number_of_clusters = len(np.unique(aff_clusters))

c_val = aff_clusters.astype(float)

marker = ".so+xd^1hX"

for i,marker_shape,c_v in zip(range(number_of_clusters),marker,c_val):

  d = data[aff_clusters==i]

  ax.set_zticks([0,0.5,1])



  ax.scatter3D(d['Age'],d['Spending Score (1-100)'],d['Genre'],marker=marker_shape,s=50)

  ax.scatter3D(aff_cc[:,1],aff_cc[:,2],aff_cc[:,0],s=200,color='black',c=None,marker="*")

plt.show()

distance = []

total = 10

for k in range(1,total):

  kmean = KMeans(n_clusters=k,init = 'k-means++')

  kmean.fit(data)

  distance.append(kmean.inertia_)

plt.plot(range(1,total), distance)

plt.title('Elbow Plot')

plt.xlabel('No. of clusters')

plt.ylabel('Distance')

plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++').fit(data.values)



k_cc = kmeans.cluster_centers_

k_clusters = kmeans.labels_

print(np.unique(k_clusters))

print(k_cc)



plt.figure(figsize=(15,10))

ax = plt.axes(projection="3d")

ax.scatter3D(data['Age'],data['Spending Score (1-100)'],data['Genre'],c=k_clusters.astype(float),s=50)

ax.scatter3D(k_cc[:,1],k_cc[:,2],k_cc[:,0],s=200,color='black',c=None,marker="*")

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score (1-100)')

ax.set_zlabel('Gender')

ax.set_zticks([0,0.5,1])

plt.show()

plt.figure(figsize=(15,10))

ax = plt.axes(projection="3d")

ax.set_xlabel('Age')

ax.set_ylabel('Spending Score (1-100)')

ax.set_zlabel('Gender')



number_of_clusters = len(np.unique(k_clusters))

c_val = k_clusters.astype(float)

marker = ".so+xd^1hX"

for i,marker_shape,c_v in zip(range(number_of_clusters),marker,c_val):

  d = data[k_clusters==i]

  ax.set_zticks([0,0.5,1])



  ax.scatter3D(d['Age'],d['Spending Score (1-100)'],d['Genre'],marker=marker_shape,s=50)

  ax.scatter3D(k_cc[:,1],k_cc[:,2],k_cc[:,0],s=200,color='black',c=None,marker="*")

plt.show()