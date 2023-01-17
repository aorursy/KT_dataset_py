import pandas as pd
df = pd.read_csv("../input/usarrests/USArrests.csv").copy()
df.head()
df.index=df.iloc[:,0]
df.index
df= df.iloc[:,1:5]
df.head(3)
df.index.name = None
df.head(2)
df.isnull().sum()
df.info()
df.describe().T
df.hist(figsize=(10,10));
#Assault has 3 peaks. We wonder that which situations constitutes these peaks? 
#This question's answers will be segmentations...
#K-Means Model and Visualizations
#set the model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
kmeans
#important parameters
#n_clusters
#Fit the model
kfit=kmeans.fit(df)
kfit.n_clusters
kfit.cluster_centers_
df.shape
kfit.labels_  #clusters labels
#Visualization 2_dimensional
kfit=KMeans(n_clusters=2).fit(df)
clustersnames=kfit.labels_
import matplotlib.pyplot as plt
plt.scatter(df.iloc[:,0],df.iloc[:,1], c=clustersnames, s=100, cmap="viridis");
centers=kfit.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c="black",s=200,alpha=0.8);
#Visualization 3_dimensional
from mpl_toolkits.mplot3d import Axes3D
kfit=KMeans(n_clusters=3).fit(df)
clusters= kfit.labels_
centers=kfit.cluster_centers_
plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]);
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=clusters)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
           marker='*', 
           c='#050505', 
           s=1000);
#clusters and observations
kfit=KMeans(n_clusters=3).fit(df)
clusters= kfit.labels_
pd.DataFrame({"States": df.index, "Clusters": clusters})[0:10]
df["cluster numbers"]=clusters
df.head()
df["cluster numbers"]=clusters+1
df.head()
#We tried k=2 and k=3, till here.
#But what is the optimum k?
#We want to maximize similarity btw clusters, minimize similarity within clusters
#we will calculate SSD
!pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,50))
visualizer.fit(df) 
visualizer.poof()  
#if we want to segment the data, the number of segmentation varies based on the line of business.
#In general, number of segmentation range is 1:5. 
# In this example, 3 is better than 2. However there isn't so much differences btw 4-5-6,so 4 is more preferable.

# we want min distortion score also
kmeans = KMeans(n_clusters = 4)
k_fit = kmeans.fit(df)
clusters = k_fit.labels_
pd.DataFrame({"States" : df.index, "Clusters": clusters})[0:10]
