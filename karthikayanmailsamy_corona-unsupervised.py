import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
corona_pd=pd.read_csv("../input/covid19-symptoms-checker/Cleaned-Data.csv")
corona_pd.sample(5)
#Returns the  meta data of the dataset.
corona_pd.info()
#Returns the information like mean,max,min,etc., of the dataset.
corona_pd.describe()
#To remove the columns of the DataFrame in memory.
corona_pd.drop(["Country"],axis=1,inplace=True)
corona_pd.sample(5)
#Returns the sum of null values under each column.
corona_pd.isnull().sum()
#To check whether the row contains duplicate values or not.
corona_pd.duplicated()
#To plot a correlation matrix between features.
f,ax= plt.subplots(figsize=(30,30))
sns.heatmap(corona_pd.corr(),annot=True)
#To scale the values along columns.
scaler= StandardScaler()
corona_pd_scaled=scaler.fit_transform(corona_pd)
#To get the Within Cluster Sum of Squares(WCSS) for each cluster count to find the optimal K value(i.e cluster count).
scores=[]
for i in range(1,20):
    corona_means=KMeans(n_clusters=i)
    corona_means.fit(corona_pd_scaled)
    scores.append(corona_means.inertia_)
#Plotting the values obtained to get the optimal K-value.
plt.plot(scores,"-rx")
#Applying K-means algorithm with the obtained K value.
corona_means=KMeans(n_clusters=7)
corona_means.fit(corona_pd_scaled)
#Returns an array with cluster labels to which it belongs.
labels=corona_means.labels_
#Creating a Dataframe with cluster centres(The example which is taken as center for each cluster)-If you are not familiar ,learn about k-means through the link given at last.
corona_pd_m=pd.DataFrame(corona_means.cluster_centers_,columns=corona_pd.columns)
corona_pd_m
#Concatenating the cluster labels.
corona_cluster=pd.concat([corona_pd,pd.DataFrame({"Cluster":labels})],axis=1)
corona_cluster.sample(5)
#Implementing pca with 3 components i.e 3d plot
corona_pca=PCA(n_components=3)
principal_comp=corona_pca.fit_transform(corona_pd_scaled)
principal_comp=pd.DataFrame(principal_comp,columns=['pca1','pca2','pca3'])
principal_comp.head()
principal_comp=pd.concat([principal_comp,pd.DataFrame({"Cluster":labels})],axis=1)
principal_comp.sample(5)
#Plotting the 2d-plot.
plt.figure(figsize=(10,10))
ax=sns.scatterplot(x='pca1',y='pca2',hue="Cluster",data=principal_comp ,palette=['red','green','blue','orange','black','yellow','violet'])
plt.show()
#Plotting the 3d-plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
sc=ax.scatter(xs=principal_comp['pca1'],ys=principal_comp['pca3'],zs=principal_comp['pca2'],c=principal_comp['Cluster'],marker='o',cmap="gist_rainbow")
plt.colorbar(sc)
plt.show()