import numpy as np 

import pandas as pd

df = pd.read_csv('../input/Data01.csv')

df.head()
df.shape
df.dtypes
df.isnull().values.any()
import seaborn as sns



sns.boxplot(x='Gender',y='Annual Income (k$)',data=df,palette='rainbow')
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=df, palette='coolwarm')
sns.jointplot(x='Age', y='Annual Income (k$)', data=df)
sns.jointplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
sns.jointplot(x='Age', y='Spending Score (1-100)', data=df)
df.drop(['CustomerID'], axis=1, inplace=True)

df.sample(4)
df.drop(['Gender'], axis=1, inplace=True)

df.sample(4)
from scipy.stats import zscore

df_scaled = df.apply(zscore)
from sklearn.cluster import KMeans



cluster_range = range(1,15)

cluster_inertia = []

for num_clusters in cluster_range:

    model = KMeans(num_clusters)

    model.fit(df_scaled)

    cluster_inertia.append(model.inertia_)
#elbow method



import matplotlib.pyplot as plt

%matplotlib inline





plt.figure(figsize=(12,6))

plt.plot(cluster_range, cluster_inertia, marker='o')
kmeans = KMeans(n_clusters=3, n_init=15, random_state=2)

kmeans.fit(df_scaled)
#Find the centroids of the clusters



centroids = kmeans.cluster_centers_

centroids
#Create a separate dataframe with the centroids



centroid_df = pd.DataFrame(centroids, columns=list(df_scaled))

centroid_df
L = pd.DataFrame(kmeans.labels_)

L[0].value_counts()
#Find out the inertia



kmeans.inertia_
#Assign labels



df_with_labels = df_scaled.copy(deep=True)

df_with_labels['labels'] = kmeans.labels_
df0 = df_with_labels[df_with_labels['labels']==0]

df0.head()
df1 = df_with_labels[df_with_labels['labels']==1]

df1.head()
df2 = df_with_labels[df_with_labels['labels']==2]

df2.head()
c0 = kmeans.cluster_centers_[0]

c1 = kmeans.cluster_centers_[1]

c2 = kmeans.cluster_centers_[2]
import numpy as np

i0=0

for i in np.arange(df0.shape[0]):

    i0= i0 + np.sum((df0.iloc[i,:-1]-c0)**2)

print(i0)
import numpy as np

i1=0

for i in np.arange(df1.shape[0]):

    i1= i1 + np.sum((df1.iloc[i,:-1]-c1)**2)

print(i1)
import numpy as np

i2=0

for i in np.arange(df2.shape[0]):

    i2= i2 + np.sum((df2.iloc[i,:-1]-c2)**2)

print(i2)
#sum of the above should equate to the inertia



i0+i1+i2
#Plot a 3D KMeans clustering plot



import numpy as np



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))

ax = Axes3D(fig, rect=[0,0,1,1], elev=70, azim=200)

labels = kmeans.labels_

ax.scatter(df_scaled.iloc[:,1], df_scaled.iloc[:,0], df_scaled.iloc[:,2], c=labels.astype(np.float), edgecolor='k')

ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('Annual Income (k$)')

ax.set_ylabel('Age')

ax.set_zlabel('Spending Score (1-100)')

ax.set_title('3D plot of KMeans Clustering')