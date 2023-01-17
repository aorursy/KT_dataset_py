# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

sns.set(style="whitegrid", color_codes=True)
df = pd.read_csv('../input/Country-data.csv')
df.head()
# Check duplicates

df.duplicated().sum()
# Checking null values

df.info()
df.describe(percentiles=[.25,.5,.75,.90,.95,.99])
# Checking null

df.isnull().sum().max()
df.set_index('country', inplace = True)
plt.figure(figsize = (25,25))

sns.pairplot(df)
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
df_log = np.log(df.drop('inflation', axis =1)) # Taking log for all features except inflation - since that has non-zero values
df_log = pd.concat([df_log, df['inflation']], axis =1 )
df_log.describe()
plt.figure(figsize = (15,15))

sns.pairplot(df_log)
plt.figure(figsize=(20, 15))

for i, x_var in enumerate(df_log.columns):

    plt.subplot(3,3,i+1)

    sns.boxplot(x = x_var, data = df_log, orient = 'v')
df_no_outliers = df_log.copy() # df for outliers removal 
for i, var in enumerate(df_no_outliers.columns):

    Q1 = df_no_outliers[var].quantile(0.25)

    Q3 = df_no_outliers[var].quantile(0.75)

    IQR = Q3 - Q1

    df_no_outliers = df_no_outliers[(df_no_outliers[var] >= Q1 - 2*IQR) & (df_no_outliers[var] <= Q3 + 2*IQR)]
df_no_outliers.describe(percentiles=[.25,.5,.75,.90,.95,.99])
sns.pairplot(df_no_outliers)
df_no_outliers.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_no_outliers), columns = df_no_outliers.columns, 

                         index=df_no_outliers.index)
df_scaled.describe()
df_scaled.head(5)
#Improting the PCA module

from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=42)
pca.fit(df_scaled)
#Making the screeplot

plt.figure(figsize = (8,6))

sns.lineplot(y = np.cumsum(pca.explained_variance_ratio_), x = range(1,len(df_scaled.columns)+1))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')
pca_again = PCA(0.95)
df_pca = pca_again.fit_transform(df_scaled)

df_pca.shape
df_pca = pd.DataFrame(df_pca, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index = df_scaled.index)
df_pca.head()
pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':list(df_scaled.columns)})

pcs_df.head(10)
fig = plt.figure(figsize = (8,8))

plt.scatter(pcs_df.PC1, pcs_df.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(pcs_df.Feature):

    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))

plt.tight_layout()

plt.show()
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
print("DF_PCA: ", hopkins(df_pca))

print("DF_scaled: ", hopkins(df_scaled))
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sse = []

for k in range(2, 20):

    kmeans = KMeans(n_clusters=k, n_init = 50).fit(df_pca)

    sse.append([k, silhouette_score(df_pca, kmeans.labels_)])

sns.pointplot(pd.DataFrame(sse)[0], y = pd.DataFrame(sse)[1])

plt.xlabel('No. of clusters')

plt.ylabel('Silhouette score')
# sum of squared distances

ssd = []

for num_clusters in list(range(1,21)):

    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)

    model_clus.fit(df_pca)

    ssd.append([num_clusters, model_clus.inertia_])



sns.pointplot(x = pd.DataFrame(ssd)[0], y = pd.DataFrame(ssd)[1])

plt.xlabel('No. of clusters')

plt.ylabel('Sum of squared errors')
from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
plt.figure(figsize=(12,8))

mergings_s = linkage(df_pca, method = "single", metric='euclidean')

dendrogram(mergings_s, labels=df_pca.index, leaf_rotation=90, leaf_font_size=6)

plt.show()
plt.figure(figsize=(12,8))

mergings_c = linkage(df_pca, method = "complete", metric='euclidean')

dendrogram(mergings_c, labels=df_pca.index, leaf_rotation=90, leaf_font_size=6)

plt.show()
plt.figure(figsize=(12,8))

mergings_a = linkage(df_pca, method = "average", metric='euclidean')

dendrogram(mergings_a, labels=df_pca.index, leaf_rotation=90, leaf_font_size=6)

plt.show()
# Kmeans with K=4

K_means_4 = KMeans(n_clusters = 4, max_iter=50)

K_means_4.fit(df_pca)
# Combining original data, principal components, K-means cluster IDs & Hierarchical clustering cluster IDs
df_2 = df_pca.merge(df, on = 'country')
df_2.head()
K_cluster_labels_4 = pd.Series(K_means_4.labels_, index = df_pca.index) # Merging data with K-mean clusterID data
df_3 = pd.concat([df_2, K_cluster_labels_4], axis = 1)
df_3.columns
df_3 = df_3.rename(columns={0: 'K_clust_4'})
df_3.head()
# Merging data with hierarchical clusterID data

# Cutting hierarchical dendogram with 4 clusters 

H_cluster_labels_4 = pd.Series(cut_tree(mergings_c, n_clusters = 4).reshape(-1,), index = df_pca.index)
df_f = pd.concat([df_3, H_cluster_labels_4], axis =1)

df_f = df_f.rename(columns={0: 'H_clust_4'})
df_f.head()
df_f.K_clust_4.value_counts()
df_f.H_clust_4.value_counts()
ct = pd.crosstab(df_f['H_clust_4'], df_f['K_clust_4'])

print(ct)
df_f.columns
plt.figure(figsize = (12,8))

sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'K_clust_4', data = df_f, palette = 'RdYlGn')
cluster_summary = df_f.groupby('K_clust_4').agg({'child_mort': 'mean', 'exports': 'mean', 'health': 'mean',

                               'imports': 'mean', 'income': 'mean', 'inflation': 'mean', 

                             'life_expec': 'mean', 'total_fer': 'mean', 'gdpp': ['mean', 'count']}).round(0)

cluster_summary
population_avg = df_f.drop(['K_clust_4', 'H_clust_4'], axis = 1).mean()
cluster_avg_4 = df_f.drop('H_clust_4', axis =1).groupby('K_clust_4').mean()
relative_imp_4 = cluster_avg_4/ population_avg - 1
relative_imp_4
plt.figure(figsize = (15,5))

PCs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

ax = sns.heatmap(relative_imp_4.drop(PCs, axis = 1), annot = True, fmt = '.2f', vmax = 1, vmin = -1, cmap = 'RdYlGn')

ax.tick_params(labelsize=13)
plt.figure(figsize = (15,5))

PCs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

ax = sns.heatmap(relative_imp_4[PCs], annot = True, cmap = 'RdYlGn')

ax.tick_params(labelsize=13)
df_sp = df_scaled.copy()

df_sp['K_clust_4'] = K_cluster_labels_4
df_sp.head()
df_melt = pd.melt(df_sp.reset_index(), id_vars = ['country','K_clust_4'], var_name = 'Feature', value_name = 'Value')
df_melt.head()
plt.figure(figsize = (12,6))

sns.lineplot(x = 'Feature', y = 'Value', data = df_melt, hue = 'K_clust_4', palette = "RdYlBu")
sns.barplot(x = cluster_summary.reset_index().K_clust_4, y = cluster_summary['gdpp']['mean'])
sns.barplot(x = cluster_summary.reset_index().K_clust_4, y = cluster_summary['income']['mean'])
sns.barplot(x = cluster_summary.reset_index().K_clust_4, y = cluster_summary['child_mort']['mean'])
cluster_summary_PC = df_f.groupby('K_clust_4').agg({'PC1': 'mean', 'PC2': 'mean', 'PC3': 'mean',

                               'PC4': 'mean', 'PC5': ['mean', 'count']}).round(2)

cluster_summary_PC
sns.barplot(x = cluster_summary_PC.reset_index().K_clust_4, y = cluster_summary_PC['PC1']['mean'])
sns.barplot(x = cluster_summary_PC.reset_index().K_clust_4, y = cluster_summary_PC['PC2']['mean'])
# Kmeans with K=3

K_means_3 = KMeans(n_clusters = 3, max_iter=50)

K_means_3.fit(df_pca)
K_cluster_labels_3 = pd.Series(K_means_3.labels_, index = df_pca.index) # Merging data with K-mean clusterID data

df_f = pd.concat([df_f, K_cluster_labels_3], axis = 1)

df_f = df_f.rename(columns={0: 'K_clust_3'})

df_f.K_clust_3.value_counts()
ct = pd.crosstab(df_f['K_clust_4'], df_f['K_clust_3'])

print(ct)
plt.figure(figsize = (12,8))

#cmap = sns.cubehelix_palette(dark=.9, light=.3, as_cmap=True)

sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'K_clust_3', data = df_f, palette= 'Set1')

#,  = 'flatui')

cluster_summary_3 = df_f.groupby('K_clust_3').agg({'child_mort': 'mean', 'exports': 'mean', 'health': 'mean',

                               'imports': 'mean', 'income': 'mean', 'inflation': 'mean', 

                             'life_expec': 'mean', 'total_fer': 'mean', 'gdpp': ['mean', 'count']}).round(0)

cluster_summary_3
population_avg = df_f.drop(['K_clust_4', 'H_clust_4', 'K_clust_3'], axis = 1).mean()

cluster_avg_3 = df_f.drop(['H_clust_4', 'K_clust_4'], axis =1).groupby('K_clust_3').mean()

relative_imp_3 = cluster_avg_3/ population_avg - 1

relative_imp_3
plt.figure(figsize = (15,5))

ax = sns.heatmap(relative_imp_3.drop(PCs, axis = 1), annot = True, fmt = '.2f', vmax = 1, vmin = -1, cmap = 'RdYlGn')

ax.tick_params(labelsize=13)
df_sp_3 = df_scaled.copy()

df_sp_3['K_clust_3'] = K_cluster_labels_3

df_melt_3 = pd.melt(df_sp_3.reset_index(), id_vars = ['country','K_clust_3'], var_name = 'Feature', value_name = 'Value')

df_melt_3.head()
plt.figure(figsize = (12,6))

sns.lineplot(x = 'Feature', y = 'Value', data = df_melt_3, hue = 'K_clust_3', palette = "Set1")
plt.figure(figsize = (12,6))

sns.lineplot(x = 'Feature', y = 'Value', data = df_melt, hue = 'K_clust_4', palette = "RdYlBu")