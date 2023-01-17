from IPython.display import Image 
Image("../input/kmeans/kmeans.png")
from warnings import filterwarnings

filterwarnings("ignore")

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp

from sklearn.cluster import KMeans
df = pd.read_csv("../input/usarrests/USArrests.csv").copy()

df.head()
# Making Urban Names İndex

df.index = df.iloc[:,0]
df= df.iloc[:,1:5]
del df.index.name
df.head()
df.isnull().sum()
df.info()
df.describe().T
df.hist(bins=15,figsize=(10,10))
kmeans = KMeans(n_clusters=4)
print(kmeans) # n_clusters: Cluster Numbers - n_init: Center Names(How many times will we fit?)
k_fit= kmeans.fit(df)
k_fit.n_clusters
# Centers of Clusters for each feature

k_fit.cluster_centers_
# Which urban in whick class

k_fit.labels_
kmeans= KMeans(n_clusters=2)

k_fit = kmeans.fit(df)
clusters = k_fit.labels_
plt.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3],)
plt.scatter(df.iloc[:,0],df.iloc[:,1],c=clusters,s=50,cmap="viridis") # First 2 feature

centers = k_fit.cluster_centers_

plt.scatter(centers[:,0],centers[:,1],c="black",s = 200,alpha=0.5)
from mpl_toolkits.mplot3d import Axes3D

# !pip install --upgrade matplotlib

# import mpl_toolkits
kmeans= KMeans(n_clusters=3)

k_fit =kmeans.fit(df)

clusters= k_fit.labels_

centers = kmeans.cluster_centers_
plt.rcParams["figure.figsize"] =(16,9)

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])

ax.scatter(centers[:,0],centers[:,1],centers[:,2],c="black",marker="*",s = 1000);
kmeans= KMeans(n_clusters=4)

k_fit =kmeans.fit(df)

clusters= k_fit.labels_
pd.DataFrame({"Urbans": df.index,"Clusters": clusters})[:10]
## Adding to DF

df["Clusters"] = clusters

df.head()
df.Clusters.unique()
df["Clusters"]=df["Clusters"]+1
df.head(10)
#!pip install yellowbrick

from yellowbrick.cluster import KElbowVisualizer

kmeans= KMeans()

visualizer = KElbowVisualizer(kmeans, k=(2,50)) # k: Number of cluster to be attempted

visualizer.fit(df)

visualizer.poof()
kmeans= KMeans(n_clusters=9)

k_fit =kmeans.fit(df)

clusters= k_fit.labels_
pd.DataFrame({"Urbans": df.index,"Clusters": clusters})[:10]