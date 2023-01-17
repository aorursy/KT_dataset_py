# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from warnings import filterwarnings

filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

from sklearn.cluster import KMeans

import mpl_toolkits

from mpl_toolkits.mplot3d import Axes3D

import sklearn

from scipy.cluster.hierarchy import linkage,dendrogram

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from yellowbrick.cluster import silhouette_visualizer

import yellowbrick

from yellowbrick.cluster import KElbowVisualizer

from yellowbrick.features import PCA

from sklearn.metrics import silhouette_score

from yellowbrick.cluster import InterclusterDistance
data=pd.read_csv("/kaggle/input/USArrests.csv")

df=data.copy()

df.head()
df.index=df.iloc[:,0]
df.index
df=df.iloc[:,1:5]
df.index.name=None

df.head()
df.isnull().sum()
df.info()
df.describe().T
df.hist(figsize=(12,5));
sns.pairplot(df);
kmeans=KMeans(n_clusters=4,random_state=42)

kmeans
kmeans.get_params()
k_fit=kmeans.fit(df)
visualizer = InterclusterDistance(kmeans)



visualizer.fit(df)        

visualizer.show() ;
silhouette_score(df,labels=kmeans.labels_)
k_fit.inertia_ 
k_fit.cluster_centers_
k_fit.n_clusters
k_fit.labels_
plt.scatter(df.iloc[:,0],df.iloc[:,1],c=k_fit.labels_);
centers=k_fit.cluster_centers_

plt.scatter(df.iloc[:,0],df.iloc[:,1],c=k_fit.labels_);

plt.scatter(centers[:,0],centers[:,1],c="red",s=200,alpha=0.5);
silhouette_visualizer(kmeans, df, colors='yellowbrick');
fig=plt.figure()

ax=Axes3D(fig)

ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=k_fit.labels_);

ax.scatter(centers[:,0],centers[:,1],centers[:,2],

           marker=".",

           c="red",

           s=1000);
kmeans=KMeans(n_clusters=2,random_state=42)

k_fit=kmeans.fit(df)

centers=k_fit.cluster_centers_

plt.scatter(df.iloc[:,0],df.iloc[:,1],c=k_fit.labels_);

plt.scatter(centers[:,0],centers[:,1],c="red",s=200,alpha=0.5);
visualizer = InterclusterDistance(kmeans)



visualizer.fit(df)        

visualizer.show() ;
silhouette_score(df,labels=kmeans.labels_)
k_fit.inertia_ 
silhouette_visualizer(kmeans, df, colors='yellowbrick');
fig=plt.figure()

ax=Axes3D(fig)

ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=k_fit.labels_);

ax.scatter(centers[:,0],centers[:,1],centers[:,2],

           marker=".",

           c="red",

           s=1000);
kmeans=KMeans(n_clusters=3,random_state=42)

k_fit=kmeans.fit(df)

centers=k_fit.cluster_centers_

plt.scatter(df.iloc[:,0],df.iloc[:,1],c=k_fit.labels_);

plt.scatter(centers[:,0],centers[:,1],c="red",s=200,alpha=0.5);
visualizer = InterclusterDistance(kmeans)



visualizer.fit(df)        

visualizer.show() ;
silhouette_score(df,labels=kmeans.labels_)
k_fit.inertia_ 
silhouette_visualizer(kmeans, df, colors='yellowbrick');
fig=plt.figure()

ax=Axes3D(fig)

ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=k_fit.labels_);

ax.scatter(centers[:,0],centers[:,1],centers[:,2],

           marker=".",

           c="red",

           s=1000);
pd.DataFrame({"States":df.index,"Clusters":k_fit.labels_})[:10]
kmeans=KMeans()

visualizer=KElbowVisualizer(kmeans,k=(2,20))

visualizer.fit(df)

visualizer.poof();
kmeans=KMeans(n_clusters=4)

k_fit=kmeans.fit(df)

pd.DataFrame({"States":df.index,"Clusters":k_fit.labels_})[:10]
visualizer = InterclusterDistance(kmeans)



visualizer.fit(df)        

visualizer.show() ;
hc_complete=linkage(df,"complete")

hc_avg=linkage(df,"average")

hc_single=linkage(df,"single")
hc_complete[:10]
hc_avg[:10]
hc_single[:10]
plt.figure(figsize=(20,10));

plt.title("Hierarchical Clustering")

plt.xlabel("index")

plt.ylabel("distance")

dendrogram(hc_complete,leaf_font_size=10);
plt.figure(figsize=(20,10));

plt.title("Hierarchical Clustering")

plt.xlabel("index")

plt.ylabel("distance")

dendrogram(hc_complete,

          truncate_mode="lastp",

          p=12,

          show_contracted=True);
plt.figure(figsize=(20,10));

plt.title("Hierarchical Clustering")

plt.xlabel("index")

plt.ylabel("distance")

dendrogram(hc_complete,

          truncate_mode="lastp",

          p=4,

          show_contracted=True);
cluster=AgglomerativeClustering(n_clusters=4,

                               affinity="euclidean",

                               linkage="ward")
cluster.fit_predict(df)
pd.DataFrame({"States":df.index,"Clusters":cluster.fit_predict(df)})[:10]
df=StandardScaler().fit_transform(df)
visualizer = PCA(scale=True, proj_features=True)

visualizer.fit_transform(df)

visualizer.show();
visualizer = PCA(scale=True, proj_features=True, projection=2,heatmap=True)

visualizer.fit_transform(df)

visualizer.show();
visualizer = PCA(scale=True, proj_features=True, projection=3)

visualizer.fit_transform(df)

visualizer.show();
pca=sklearn.decomposition.PCA(n_components=2)

pca_fit=pca.fit_transform(df)
pca_fit[:10]
pca_df=pd.DataFrame(pca_fit,

             columns=["comp_1","comp_2"])

pca_df.head()
pca.explained_variance_ratio_
pca.explained_variance_
pca=sklearn.decomposition.PCA(n_components=3)

pca_fit=pca.fit_transform(df)
pca_fit[:5]
pca_df=pd.DataFrame(pca_fit,

             columns=["comp_1","comp_2","comp_3"])

pca_df.head()
pca.explained_variance_ratio_
pca=sklearn.decomposition.PCA().fit(df)

plt.plot(np.cumsum(pca.explained_variance_ratio_));