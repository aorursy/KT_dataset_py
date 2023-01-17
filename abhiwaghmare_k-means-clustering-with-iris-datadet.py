import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/iris/Iris.csv',index_col=0)
df.drop('Species',axis=1,inplace=True)

df.head()
df.info
df.isnull().sum()
x = df.values
plt.figure(figsize=(10,6))
plt.scatter(x[:,0],x[:,1])
plt.figure(figsize=(10,6))
plt.scatter(x[:,2],x[:,3])
from sklearn.cluster import KMeans
wcss =[]
for i in range(1,11):
    k_means = KMeans(n_clusters = i,init = 'k-means++',random_state = 0)
    k_means.fit(x)
    wcss.append(k_means.inertia_)
print(wcss)
x_range = range(1,11)
plt.plot(x_range,wcss)
k_means = KMeans(n_clusters = 3,init = 'k-means++',random_state = 0)
k_means.fit(x)
y_kmeans=k_means.predict(x)
y_kmeans
plt.figure(figsize=(10,6))
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='magenta',label='cluster3')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=200,c='black',label='cluster_centers')
plt.legend()
plt.show()
plt.figure(figsize=(10,6))
plt.scatter(x[y_kmeans==0,2],x[y_kmeans==0,3],s=100,c='red',label='cluster1')
plt.scatter(x[y_kmeans==1,2],x[y_kmeans==1,3],s=100,c='blue',label='cluster2')
plt.scatter(x[y_kmeans==2,2],x[y_kmeans==2,3],s=100,c='magenta',label='cluster3')
plt.scatter(k_means.cluster_centers_[:,2],k_means.cluster_centers_[:,3],s=200,c='black',label='cluster_centers')
plt.legend()
plt.show()
