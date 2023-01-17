import pandas as pd

import numpy as np

import pandas as pd



# For Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# To Scale our data

from sklearn.preprocessing import scale



# To perform KMeans clustering 

from sklearn.cluster import KMeans



# To perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
import warnings

warnings.filterwarnings('ignore')
#importing Dataset already uploaded to ANACONDA cloud

df_original = pd.read_csv('../input/Country-data.csv')

df = pd.read_csv('../input/Country-data.csv')
df.head()
df.shape
df.info()
#Checking outliers for ciontinuous variables

# Checking outliers at 25%,50%,75%,90%,95% and 99%

df.describe(percentiles=[.25,.50,.75,.90,.95,.99])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y= df.pop('country')
X = df.copy()
x = scaler.fit_transform(X)
x
from sklearn.decomposition import PCA
pca = PCA(svd_solver = 'randomized', random_state=42)
pca.fit(x)
pca.components_
pca.explained_variance_ratio_
plt.ylabel('Cumulative Variance')

plt.xlabel('Number of Components')

plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
var_cumu = np.cumsum(pca.explained_variance_ratio_)

var_cumu
plt.plot(range(1,(len(var_cumu)+1)),var_cumu)

plt.title('Scree Plot')

plt.xlabel('Cumulative Variance')

plt.ylabel('Number of Components')
from sklearn.decomposition import IncrementalPCA
pc5 = PCA(n_components = 5, random_state=42)
df_pc5 = pc5.fit_transform(x)
df_pc5.shape
df_pc5[:10]
df5 = pd.DataFrame(df_pc5, columns=['PC1','PC2','PC3','PC4','PC5'])
df5.head()
df5_final = pd.concat([df5,y], axis=1)
df5_final.head()
plt.figure(figsize=(20,5))

# For PC1

plt.subplot(1,5,1)

plt.title('PC1')

plt.boxplot(df5_final.PC1)



Q1 = df5_final.PC1.quantile(0.05)

Q3 = df5_final.PC1.quantile(0.95)

IQR = Q3-Q1

df5_final = df5_final[(df5_final.PC1>=Q1) & (df5_final.PC1<=Q3)]



# For PC2

plt.subplot(1,5,2)

plt.title('PC2')

plt.boxplot(df5_final.PC2)



Q1 = df5_final.PC2.quantile(0.05)

Q3 = df5_final.PC2.quantile(0.95)

IQR = Q3-Q1

df5_final = df5_final[(df5_final.PC2>=Q1) & (df5_final.PC2<=Q3)]





# For PC3

plt.subplot(1,5,3)

plt.title('PC3')

plt.boxplot(df5_final.PC3)



Q1 = df5_final.PC3.quantile(0.05)

Q3 = df5_final.PC3.quantile(0.95)

IQR = Q3-Q1

df5_final = df5_final[(df5_final.PC3>=Q1) & (df5_final.PC3<=Q3)]





# For PC4

plt.subplot(1,5,4)

plt.title('PC4')

plt.boxplot(df5_final.PC4)



Q1 = df5_final.PC4.quantile(0.05)

Q3 = df5_final.PC4.quantile(0.95)

IQR = Q3-Q1

df5_final = df5_final[(df5_final.PC4>=Q1) & (df5_final.PC4<=Q3)]





# For PC5

plt.subplot(1,5,5)

plt.title('PC5')

plt.boxplot(df5_final.PC5)



Q1 = df5_final.PC5.quantile(0.05)

Q3 = df5_final.PC5.quantile(0.95)

IQR = Q3-Q1

df5_final = df5_final[(df5_final.PC5>=Q1) & (df5_final.PC5<=Q3)]
### Clustering - Calculating the Hopkins statistic#Calculating the Hopkins statistic

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
hopkins(df5_final.drop('country', axis =1))
df6_final = df5_final.drop('country', axis = 1)
#First we'll do the silhouette score analysis

from sklearn.metrics import silhouette_score



ss = []

for k in range(2,10):

    kmeans = KMeans(n_clusters = k).fit(df6_final)

    ss.append([k,silhouette_score(df6_final, kmeans.labels_)])
plt.title('Silhoette')

plt.plot(pd.DataFrame(ss)[0], pd.DataFrame(ss)[1]);
#Now let's proceed to the elbow curve method

from sklearn.metrics import silhouette_score



ssd = []

for k in range(1,10):

    model = KMeans(n_clusters = k, max_iter = 50).fit(df6_final)

    ssd.append([model.inertia_])



print(ssd)
plt.title('Elbow')

plt.plot(ssd)
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(df6_final)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(df6_final, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
#Let's perform K means using K=

model_clus2 = KMeans(n_clusters = 3, max_iter = 50, random_state = 50)

model_clus2.fit(df6_final)
# Let's add the cluster Ids to the PCs data 



dat_km = pd.concat([df5_final.reset_index().drop('index', axis=1), pd.Series(model_clus2.labels_)], axis =1)
dat_km.head()
dat_km.columns = ['PC1', 'PC2', 'PC3','PC4','PC5', 'country','ClusterID']

dat_km.head()
# Check the count of observation per cluster

dat_km['ClusterID'].value_counts()
# Plot the Cluster with respect to the clusters obtained



plt.figure(figsize=(15,4))



plt.subplot(1,3,1)

sns.scatterplot(x='PC1', y ='PC2', hue = 'ClusterID', palette=['green','dodgerblue','red'], legend='full', data = dat_km)

plt.subplot(1,3,2)

sns.scatterplot(x='PC1', y ='PC3', hue = 'ClusterID', palette=['green','dodgerblue','red'], legend='full', data = dat_km)

plt.subplot(1,3,3)

sns.scatterplot(x='PC2', y ='PC3', hue = 'ClusterID', palette=['green','dodgerblue','red'], legend='full', data = dat_km)

# Let's merge the original data with the data(ClusterID)

dat5 = pd.merge(df_original, dat_km, how = 'inner', on = 'country')

dat5.head()
dat5.shape
dat6 = dat5[['country','child_mort', 'income','gdpp', 'PC1', 'PC2', 'ClusterID']]
dat6.groupby('ClusterID').count()
child_mort = dat6.groupby(['ClusterID']).child_mort.mean()

income = dat6.groupby(['ClusterID']).income.mean()

gdpp = dat6.groupby(['ClusterID']).gdpp.mean()
final_df = pd.concat([child_mort, income, gdpp], axis = 1)
final_df
plt.figure(figsize=(20, 8))

plt.subplot(1,3,1)

sns.boxplot(x='ClusterID', y='income', data=dat6)



plt.subplot(1,3,2)

sns.boxplot(x='ClusterID', y='child_mort', data=dat6)



plt.subplot(1,3,3)

sns.boxplot(x='ClusterID', y='gdpp', data=dat6)
# List of Countries which need Attention.



dat6[dat6['ClusterID']==2]['country']
rfm_df = df[['child_mort', 'income','gdpp']]



# instantiate

scaler = StandardScaler()



# fit_transform

rfm_df_scaled = scaler.fit_transform(rfm_df)

rfm_df_scaled.shape
rfm_df_scaled[:10]
# single linkage

plt.figure(figsize=(20,10))

plt.title('Dendrogram - Single Linkage')

mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')

dendrogram(mergings)



plt.show()
# complete linkage

plt.figure(figsize=(20,10))

plt.title('Dendrogram - Complete Linkage')

mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')

dendrogram(mergings)

plt.show()
# 3 clusters

cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )

cluster_labels
# assign cluster labels

rfm_df_scaled = pd.DataFrame(rfm_df_scaled)

rfm_df_scaled.columns = ['child_mort', 'income','gdpp']

rfm_df_scaled.head()

rfm_df_scaled['cluster_labels'] = cluster_labels

rfm_df_scaled['country'] = df_original['country']

rfm_df_scaled.groupby('cluster_labels').count()
# Plot the Cluster with respect to the clusters obtained



plt.figure(figsize=(20,5))



plt.subplot(1,3,1)

sns.scatterplot(x='child_mort', y ='income', hue = 'cluster_labels', palette=['green','dodgerblue','red'], legend='full', data = rfm_df_scaled)

plt.subplot(1,3,2)

sns.scatterplot(x='child_mort', y ='gdpp', hue = 'cluster_labels', palette=['green','dodgerblue','red'], legend='full', data = rfm_df_scaled)

plt.subplot(1,3,3)

sns.scatterplot(x='gdpp', y ='income', hue = 'cluster_labels', palette=['green','dodgerblue','red'], legend='full', data = rfm_df_scaled)

rfm_df_scaled[rfm_df_scaled['cluster_labels']==0].head(10)
rfm_df_scaled[rfm_df_scaled['cluster_labels']==1].head(10)
rfm_df_scaled[rfm_df_scaled['cluster_labels']==2]
plt.figure(figsize=(20,8))

plt.subplot(1,3,1)

sns.boxplot(x='cluster_labels', y='income', data=rfm_df_scaled)



plt.subplot(1,3,2)

sns.boxplot(x='cluster_labels', y='child_mort', data=rfm_df_scaled)



plt.subplot(1,3,3)

sns.boxplot(x='cluster_labels', y='gdpp', data=rfm_df_scaled)
rfm_df_scaled[rfm_df_scaled['cluster_labels']==1]['country']
# List of Countries which need Attention.



df_H = rfm_df_scaled[rfm_df_scaled['cluster_labels']==1]

df_H = df_H[['country', 'child_mort','income', 'gdpp']]

df_H.head(30)
# List of Countries which need Attention.



df_K = dat6[dat6['ClusterID']==2]

df_K = df_K[['country', 'child_mort','income', 'gdpp']]

df_K.head(30)
df_combined = pd.concat([df_K, df_H], join = 'inner' )

df_combined.head(20)



df_combined.head()
plt.figure(figsize=(15,15))

plt.subplot(1,3,1)

plt.title('child_mort')

sns.barplot(x="child_mort", y="country", data=df_combined.sort_values(by=['income'], ascending = True))



plt.subplot(1,3,2)

plt.title('income')

sns.barplot(x="income", y="country", data=df_combined.sort_values(by=['income'], ascending = True))



plt.subplot(1,3,3)

plt.title('gdpp')

sns.barplot(x="gdpp", y="country", data=df_combined.sort_values(by=['income'], ascending = True))

# Bottom 20 Countries which need help

df_combined.sort_values(by=['income'], ascending = True)[:20]['country']