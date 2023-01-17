import pandas as pd

import numpy as np

from numpy import unique

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as sch

import seaborn as sns

import warnings as w

w.filterwarnings('ignore')

from sklearn.metrics import silhouette_samples, silhouette_score

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data=pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')

data.head(10)
data.shape
data.info()
data.describe()
data.isnull().sum()
data.duplicated().sum()
from sklearn.preprocessing import StandardScaler

X = StandardScaler() 

for column in data[['Genre']]:

    if data[column].dtype == 'object':

        data[column] = pd.Categorical(data[column]).codes 

data.info()
scaled_df = X.fit_transform(data.iloc[:,1:6])

scaled_df
wcss=[]

for i in range(1,11): 

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(scaled_df)

    wcss.append(kmeans.inertia_)  

plt.plot(range(1,11),wcss,marker="X",c="black")

plt.title("Elbow plot")
kmeans=KMeans(n_clusters=2)

y_kmeans=kmeans.fit(scaled_df)

kmeans.labels_
np.unique(kmeans.labels_)
centers = kmeans.cluster_centers_

centers
kmeans.inertia_
kmeans = KMeans(n_clusters = 1)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 2)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 3)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 4)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 5)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 6)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 7)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 8)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 9)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 10)

kmeans.fit(scaled_df)

kmeans.inertia_
kmeans = KMeans(n_clusters = 11)

kmeans.fit(scaled_df)

kmeans.inertia_
wss =[]

for i in range(1,11):

    KM = KMeans(n_clusters=i)

    KM.fit(scaled_df)

    wss.append(KM.inertia_)

wss
kmeans = KMeans(n_clusters = 3)

kmeans.fit(scaled_df)

labels = kmeans.labels_

silhouette_score(scaled_df,labels)
kmeans = KMeans(n_clusters = 4)

kmeans.fit(scaled_df)

labels = kmeans.labels_

silhouette_score(scaled_df,labels)
kmeans = KMeans(n_clusters = 5)

kmeans.fit(scaled_df)

labels = kmeans.labels_

silhouette_score(scaled_df,labels)
kmeans = KMeans(n_clusters = 6)

kmeans.fit(scaled_df)

labels = kmeans.labels_

silhouette_score(scaled_df,labels)
#since silhouette score is best for 5 clusters we consider 5 clusters in the segregation

kmeans = KMeans(n_clusters = 5)

kmeans.fit(scaled_df)

labels = kmeans.labels_

silhouette_score(scaled_df,labels)
data["Clus_kmeans5"] = labels

data.head()
data.Clus_kmeans5.value_counts().sort_index()
clust_profile=data.drop(['CustomerID'],axis=1)

clust_profile=clust_profile.groupby('Clus_kmeans5').mean()

clust_profile['freq']=data.Clus_kmeans5.value_counts().sort_index()

clust_profile
centers = kmeans.cluster_centers_



centers
plt.rcParams['figure.figsize'] = (8,4)

sns.heatmap(data.corr(), cmap = 'RdPu', annot = True)

sns.set_style("ticks")

plt.title('Heatmap', fontsize = 20)

plt.show()
sns.set(style="ticks", color_codes=True)

sns.pairplot(data)

plt.title('Pair-plot',fontsize = 10)

plt.show()
plt.rcParams['figure.figsize'] = (6,4)

dendrogram = sch.dendrogram(sch.linkage(scaled_df, method = 'ward'))

plt.title('Dendrogam', fontsize = 10)

plt.xlabel('Customers')

plt.ylabel('Ecuclidean Distance')

plt.show()
kmeans=KMeans(n_clusters=5)

y_kmeans=kmeans.fit(scaled_df)

plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'],c=kmeans.labels_,s=50)

plt.scatter(centers[:,0],centers[:,1],color='black',marker='s',s=100) 

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('5 Cluster K_Means')

plt.show()