import pandas as pd
import numpy as np
df=pd.read_csv("../input/ccdata/CC GENERAL.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
df.fillna(method='ffill',inplace=True)
df.head()
df.isnull().sum().sort_values(ascending=False).head()
#we dont need customer_id to classify
df.drop(["CUST_ID"],axis=1,inplace=True)
df.head()
from sklearn.preprocessing import StandardScaler,normalize
sc=StandardScaler()
df=sc.fit_transform(df)
df=normalize(df)
df=pd.DataFrame(df)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
df=pca.fit_transform(df)
#elbow method to find optimal clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
from matplotlib import pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(range(1,11),wcss)
plt.xlabel("no.of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
plt.show()
#No. of clusters is 3
kmeans=KMeans(n_clusters=3,init="k-means++")
yhat=kmeans.fit_predict(df)
plt.figure(figsize=(8,6))
plt.scatter(df[yhat==0,0],df[yhat==0,1],s=100,color="blue",label="Cluster-1")
plt.scatter(df[yhat==1,0],df[yhat==1,1],s=100,color="red",label="Cluster-2")
plt.scatter(df[yhat==2,0],df[yhat==2,1],s=100,color="green",label="Cluster-3")
plt.title("Clusters of Clients")
plt.legend()
plt.show()
# Silhouette Score
s=[]
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering 
for i in range(2,10):
    s.append(silhouette_score(df,AgglomerativeClustering(n_clusters=i).fit_predict(df)))
plt.figure(figsize=(8,6))
plt.bar(range(2,10), s)
plt.xlabel('Number of clusters') 
plt.ylabel('Silhouette Score') 
plt.show() 
y=AgglomerativeClustering(n_clusters=2).fit_predict(df)
# Visualizing the clustering 
df=pd.DataFrame(df)
plt.figure(figsize=(8,6))
plt.scatter(df.iloc[:,0], df.iloc[:,1],  
           c = y, cmap =plt.cm.winter) 
