import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.options.display.float_format = "{:.2f}".format
df = pd.read_csv("/kaggle/input/unsupervised-learning-on-country-data/Country-data.csv")
df.head()
# Changing percentage coulmns to actual values
df['exports']=df['gdpp']*df['exports']*100
df['health']=(df['gdpp']*df['health'])*100
df['imports']=(df['gdpp']*df['imports'])*100
df.head()
df.info()
df.describe()
print("shape of dataset is" ,df.shape)
df.dtypes
df.columns
print('Null values: \n{}'.format(df.isnull().sum()))
print('\nNaN values: \n{}'.format(df.isna().sum()))
plt.figure(figsize = (25,15))
sns.pairplot(df, diag_kind='kde')
plt.show()
plt.figure(figsize = (25,15))
ax = sns.heatmap(df.corr(),square = True,annot=True, cmap="Blues")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5);
fig, axs = plt.subplots(3,3,figsize = (30,30))

# Child Mortality Rate : Death of children under 5 years of age per 1000 live births
Child_mort = df[['country','child_mort']].sort_values('child_mort', ascending = False).head(5)
ax = sns.barplot(x='country', y='child_mort', data= Child_mort, ax = axs[0,0])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', title= 'Child Mortality Rate')

# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same
Total_fer = df[['country','total_fer']].sort_values('total_fer', ascending = False).head(5)
ax = sns.barplot(x='country', y='total_fer', data= Total_fer, ax = axs[0,1])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', title= 'Fertility Rate')

# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same
Life_expec = df[['country','life_expec']].sort_values('life_expec', ascending = True).head(5)
ax = sns.barplot(x='country', y='life_expec', data= Life_expec, ax = axs[0,2])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', title= 'Life Expectancy')

# Health :Total health spending.
Health = df[['country','health']].sort_values('health', ascending = True).head(5)
ax = sns.barplot(x='country', y='health', data= Health, ax = axs[1,0])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', title= 'Health')

# The GDP per capita : Calculated as the Total GDP divided by the total population.
GDPP = df[['country','gdpp']].sort_values('gdpp', ascending = True).head(5)
ax = sns.barplot(x='country', y='gdpp', data= GDPP, ax = axs[1,1])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', title= 'GDP per capita')

# Per capita Income : Net income per person
Income = df[['country','income']].sort_values('income', ascending = True).head(5)
ax = sns.barplot(x='country', y='income', data= Income, ax = axs[1,2])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', title= 'Per capita Income')


# Inflation: The measurement of the annual growth rate of the Total GDP
Inf = df[['country','inflation']].sort_values('inflation', ascending = False).head(5)
ax = sns.barplot(x='country', y='inflation', data= Inf, ax = axs[2,0])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', title= 'Inflation Rate')


# Exports: Exports of goods and services.
Exports = df[['country','exports']].sort_values('exports', ascending = True).head(5)
ax = sns.barplot(x='country', y='exports', data= Exports, ax = axs[2,1])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', title= 'Exports')


# Imports: Imports of goods and services.
Imports = df[['country','imports']].sort_values('imports', ascending = True).head(5)
ax = sns.barplot(x='country', y='imports', data= Imports, ax = axs[2,2])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', title= 'Imports')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.tight_layout()    
plt.show()
plt.figure(figsize = (25,12))
features = ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']
for i in enumerate(features):
    plt.subplot(3,3,i[0]+1)
    sns.distplot(df[i[1]])
plt.figure(figsize = (25,25))
features = ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']
for i in enumerate(features):
    plt.subplot(5,2,i[0]+1)
    sns.boxplot(df[i[1]])
# Capping of outliers
features = ['exports', 'health', 'imports', 'income','inflation', 'total_fer', 'gdpp']
for i in features:
    q1 = df[i].quantile(0.01)
    q4 = df[i].quantile(0.99)
    df[i][df[i]>=q4]=q4
plt.figure(figsize = (25,25))
features = ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']
for i in enumerate(features):
    plt.subplot(5,2,i[0]+1)
    sns.boxplot(df[i[1]])
#Calculating the Hopkins statistic
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
hopkins(df.drop('country' , axis=1))
df_scale = df[['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']]

# instantiating the scaler
scaler = StandardScaler()

# fit and transform
df_scale = scaler.fit_transform(df_scale)
df_scale.shape
df_scale = pd.DataFrame(df_scale)
df_scale.columns = ['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']
df_scale.head()
# elbow-curve/SSD
ssd = []
n_cluster = list(range(1,10))
for num_clusters in n_cluster:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df_scale)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(n_cluster,ssd,marker='o',markersize=7)
plt.vlines(x=3, ymax=ssd[-1], ymin=ssd[0], colors="g", linestyles="-")
plt.hlines(y=ssd[2], xmax=9, xmin=1, colors="r", linestyles="-")
plt.xlabel('Number of clusters',fontsize=15)
plt.ylabel('Sum of Squared distance',fontsize=15)
plt.title("Elbow Curve")
plt.show()
ss = []
for k in range(2,11):
    kmeans = KMeans(n_clusters = k).fit(df_scale)
    ss.append([k, silhouette_score(df_scale, kmeans.labels_)])

plt.plot(pd.DataFrame(ss)[0], pd.DataFrame(ss)[1],marker='o',markersize=7)
plt.xlabel('Number of Clusters',fontsize=15)
plt.ylabel('Silhouette Width',fontsize=15)
plt.title("Silhouette Score")
plt.show()
kmeans = KMeans(n_clusters=3, max_iter=50,random_state = 14)
kmeans.fit(df_scale)
kmeans.labels_
# Entering the Cluster in the column 'cluster_K' for further analysis

cluster_K = pd.DataFrame(kmeans.labels_, columns = ['cluster_K'])
# Saving the new dataframe for further analysis

df_cluster = df.copy()
# Combing the cluster with cluster labels extracted from K-means

df_cluster = pd.concat([df_cluster, cluster_K ], axis =1)
df_cluster.head()
# To check How many datapoints we have in each cluster
df_cluster.cluster_K.value_counts().reset_index()
# Scatter-plot:

f, axes = plt.subplots(1, 3, figsize=(20,5))
sns.scatterplot(x='income', y='child_mort', hue='cluster_K', data=df_cluster, palette='Set1',ax=axes[0]);
sns.scatterplot(x='gdpp', y='income', hue='cluster_K', data=df_cluster, palette='Set1',ax=axes[1]);

sns.scatterplot(x='gdpp', y='child_mort', hue='cluster_K', data=df_cluster, palette='Set1',ax=axes[2]);
# Box-plot:

f, axes = plt.subplots(1, 3, figsize=(25,7))
sns.boxplot(x='cluster_K',y='gdpp',data=df_cluster,ax=axes[0])
axes[0].set_title('GDP per capita',fontsize=15)
sns.boxplot(x='cluster_K',y='income',data=df_cluster,ax=axes[1])
axes[1].set_title('Income per person',fontsize=15)
sns.boxplot(x='cluster_K',y='child_mort',data=df_cluster,ax=axes[2])
axes[2].set_title('Child Mortality rate',fontsize=15)
plt.show()
df_clusterK = df_cluster[['child_mort','income','gdpp','cluster_K']]
df_clusterK = df_clusterK.groupby('cluster_K').mean()
df_clusterK
df_clusterK.plot(kind='bar',logy=True);
df_cluster[df_cluster['cluster_K'] ==2]['country'].reset_index(drop=True)
top5_Kmeans = df_cluster[df_cluster['cluster_K'] ==2].sort_values(by = ['income', 'gdpp', 'child_mort'], ascending = [True, True, False]).head(5)
top5_Kmeans.reset_index(drop=True)
# Taking the already scaled dataset

df_scale.head()
df_cluster.head()
# single linkage
mergings = linkage(df_scale, method="single", metric='euclidean')
dendrogram(mergings)
plt.title("Single Linkage")
plt.show()
# complete linkage
mergings = linkage(df_scale, method="complete", metric='euclidean')
dendrogram(mergings)
plt.title("Complete Linkage")
plt.show()
# 3 clusters
cluster_H = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_H
# assign cluster labels
df_cluster['cluster_H'] = cluster_H
df_cluster.head()
df_cluster.cluster_H.value_counts().reset_index()
# Scatter-Plot : 

f, axes = plt.subplots(1, 3, figsize=(20,5))
sns.scatterplot(x='income', y='child_mort', hue='cluster_H', data=df_cluster, palette='Set1',ax=axes[0]);
sns.scatterplot(x='gdpp', y='income', hue='cluster_H', data=df_cluster, palette='Set1',ax=axes[1]);
sns.scatterplot(x='gdpp', y='child_mort', hue='cluster_H', data=df_cluster, palette='Set1',ax=axes[2]);
# Boxplot :
f, axes = plt.subplots(1, 3, figsize=(20,5))
sns.boxplot(x='cluster_H', y='child_mort', data=df_cluster,ax=axes[0]);
axes[0].set_title('Child Mortality Rate',fontsize=15)
sns.boxplot(x='cluster_H', y='gdpp', data=df_cluster,ax=axes[1]);
axes[1].set_title('GDP per capita',fontsize=15)
sns.boxplot(x='cluster_H', y='income', data=df_cluster,ax=axes[2]);
axes[2].set_title('Income per person',fontsize=15)
plt.show()
df_clusterH = df_cluster[['child_mort','income','gdpp','cluster_H']]
df_clusterH = df_clusterH.groupby('cluster_H').mean()
df_clusterH
df_clusterH.plot(kind = 'bar',logy=True);
df_cluster[df_cluster['cluster_H'] ==0]['country'].reset_index(drop=True)
top5_Hier = df_cluster[df_cluster['cluster_H'] ==0].sort_values(by = ['income', 'gdpp', 'child_mort'], ascending = [True, True, False]).head(5)
top5_Hier.reset_index(drop=True)
for countries in top5_Kmeans.country:
    print(countries)