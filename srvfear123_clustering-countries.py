import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Let's start with importing the libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
#Let's read the csv file to a dataframe 'df'

df = pd.read_csv("/kaggle/input/country-dataset/Country-data.csv")
df.head()
#Checking for the data types. As we see country is a stringID column hence it is object type
df.info()
#Checking various statistics of columns like mean, median , presence of outliers
df.describe(percentiles=[0.25,0.5,0.75,0.95,0.99])
#Converting the columns 'exports','health','imports' to absolute value as they are a percentage of Gdp

df['exports'] = df['exports']*df['gdpp']/100
df['health'] = df['health']*df['gdpp']/100
df['imports'] = df['imports']*df['gdpp']/100
df.head()
#Checking for the inter-relationships of the attributes
sns.pairplot(df)
sns.heatmap(df.corr())
#Since our problem is focussed on countries who need help, we need to restrict outliers that lie in a developed nation's range. We do not cap outliers for under developed nations as we may miss out some countries that actually might require help

#low child mortality can be capped
q = df['child_mort'].quantile(0.01)
df['child_mort'][df['child_mort'] < q] = q
sns.boxplot(df['child_mort'])
plt.show()
#Capping low inflation outliers

q = df['inflation'].quantile(0.01)
df['inflation'][df['inflation'] < q] = q
sns.boxplot(df['inflation'])
plt.show()
#Capping low total_fer outliers

q = df['total_fer'].quantile(0.01)
df['total_fer'][df['total_fer'] < q] = q
sns.boxplot(df['total_fer'])
plt.show()
#Capping high income outliers to 90 percentile range. We take a little higher range because countries such income levels are self sufficient

q = df['income'].quantile(0.9)
df['income'][df['income'] > q] = q
sns.boxplot(df['income'])
plt.show()
# Similarly Capping countries of high health with 90percentile

q = df['health'].quantile(0.9)
df['health'][df['health'] > q] = q
sns.boxplot(df['health'])
plt.show()
#Same capping applies to both export and import

q = df['exports'].quantile(0.9)
df['exports'][df['exports'] > q] = q
sns.boxplot(df['exports'])
plt.show()
q = df['imports'].quantile(0.9)
df['imports'][df['imports'] > q] = q
sns.boxplot(df['imports'])
plt.show()
#Capping higher life expectancy to 90percentile range
q = df['life_expec'].quantile(0.9)
df['life_expec'][df['life_expec'] > q] = q
sns.boxplot(df['life_expec'])
plt.show()
#Overall GDP outliers capped at 90percentile range
q = df['gdpp'].quantile(0.9)
df['gdpp'][df['gdpp'] > q] = q
sns.boxplot(df['gdpp'])
plt.show()
from sklearn.preprocessing import StandardScaler
sk = StandardScaler()
scaled_df = sk.fit_transform(df.iloc[:,1:])
scaled_df
df_scaled = pd.DataFrame(scaled_df)
df_scaled.head()
# Checking for optimal number of clusters using the Elbow approach
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df_scaled)
    
    ssd.append(kmeans.inertia_)
    
plt.plot(ssd)
#Let's check the Hopkin's score

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H
#Let's take the average of 10 iteration of Hopkin's score
a=[]
for i in range(10):
    a.append(hopkins(df_scaled))

print(sum(a)/len(a))
#Let's check the Silhoutte score

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(df_scaled, kmeans.labels_)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
#Applying the Kmeans package

kmeans = KMeans(n_clusters=3, max_iter=50,random_state=100)
kmeans.fit(df_scaled)
#Merging the clustered labels into the dataframe df

df['labels'] = kmeans.labels_
df.head()
sns.boxplot(x='labels', y='child_mort', data=df)
sns.boxplot(x='labels', y='gdpp', data=df)
sns.boxplot(x='labels', y='income', data=df)
df['labels'].value_counts()
target = df[df['labels']==0]
target.sort_values(by=['child_mort','gdpp','income'],ascending=[False,True,True]).head()
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df_scaled)

df_array = pca.transform(df_scaled)

fig, (ax1) = plt.subplots(1, 1)
sns.scatterplot(x= df_array[:, 0], y=df_array[:,1], ax=ax1)
fig, (ax2) = plt.subplots(1, 1)

cmap  = sns.color_palette("Set1", n_colors=3)
sns.scatterplot(x= df_array[:, 0], y=df_array[:,1], hue=kmeans.labels_,  palette = cmap, ax=ax2)
#Lets fit the single linkage with the scaled data

mergings = linkage(df_scaled, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()
#Cutting the tree with 3 clusters

h_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
h_labels
#Adding the label column to the dataset

df['labels'] = h_labels
sns.boxplot(x='labels', y='gdpp', data=df)
df[(df['labels']==1) | (df['labels']==2)]
sns.boxplot(x='labels', y='child_mort', data=df)
sns.boxplot(x='labels', y='income', data=df)
df['labels'].value_counts()
df[df['labels']==0].sort_values(by=['child_mort','gdpp','income'],ascending=[False,True,True]).head()
mergings = linkage(df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()
h_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )  
h_labels
df['labels'] = h_labels
df
df['labels'].value_counts()
sns.boxplot(x='labels', y='gdpp', data=df)
df[df['labels']==0].sort_values(by=['child_mort','gdpp','income'],ascending=[False,True,True]).head()