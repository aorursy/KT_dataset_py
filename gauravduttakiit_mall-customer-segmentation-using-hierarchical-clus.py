# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Data display coustomization

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', -1)
# To perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
# import all libraries and dependencies for machine learning

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan
mall= pd.read_csv(r"/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

mall.head()
mall.shape
mall.info()
mall.describe()
mall_d= mall.copy()

mall_d.drop_duplicates(subset=None,inplace=True)
mall_d.shape
mall.shape
(mall.isnull().sum() * 100 / len(mall)).value_counts(ascending=False)
mall.isnull().sum()
(mall.isnull().sum(axis=1) * 100 / len(mall)).value_counts(ascending=False)
mall.isnull().sum(axis=1).value_counts(ascending=False)
plt.figure(figsize = (5,5))

gender = mall['Gender'].sort_values(ascending = False)

ax = sns.countplot(x='Gender', data= mall)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=90)

plt.show()
 

plt.figure(figsize = (20,5))

gender = mall['Age'].sort_values(ascending = False)

ax = sns.countplot(x='Age', data= mall)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))



plt.show()
plt.figure(figsize = (25,5))

gender = mall['Annual Income (k$)'].sort_values(ascending = False)

ax = sns.countplot(x='Annual Income (k$)', data= mall)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))



plt.show()
plt.figure(figsize = (27,5))

gender = mall['Spending Score (1-100)'].sort_values(ascending = False)

ax = sns.countplot(x='Spending Score (1-100)', data= mall)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))



plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (5,5))

sns.heatmap(mall.corr(), annot = True, cmap="rainbow")

plt.savefig('Correlation')

plt.show()
sns.pairplot(mall,corner=True,diag_kind="kde")

plt.show()
# Data before Outlier Treatment 

mall.describe()
f, axes = plt.subplots(1,3, figsize=(15,5))

s=sns.violinplot(y=mall.Age,ax=axes[0])

axes[0].set_title('Age')

s=sns.violinplot(y=mall['Annual Income (k$)'],ax=axes[1])

axes[1].set_title('Annual Income (k$)')

s=sns.violinplot(y=mall['Spending Score (1-100)'],ax=axes[2])

axes[2].set_title('Spending Score (1-100)')

plt.show()

Q3 = mall['Annual Income (k$)'].quantile(0.99)

Q1 = mall['Annual Income (k$)'].quantile(0.01)

mall['Annual Income (k$)'][mall['Annual Income (k$)']<=Q1]=Q1

mall['Annual Income (k$)'][mall['Annual Income (k$)']>=Q3]=Q3
# Data After Outlier Treatment 

mall.describe()
f, axes = plt.subplots(1,3, figsize=(15,5))

s=sns.violinplot(y=mall.Age,ax=axes[0])

axes[0].set_title('Age')

s=sns.violinplot(y=mall['Annual Income (k$)'],ax=axes[1])

axes[1].set_title('Annual Income (k$)')

s=sns.violinplot(y=mall['Spending Score (1-100)'],ax=axes[2])

axes[2].set_title('Spending Score (1-100)')

plt.show()
# Dropping CustomerID,Gender field to form cluster



mall_c = mall.drop(['CustomerID','Gender'],axis=1,inplace=True)
mall.head()
def hopkins(X):

    d = X.shape[1]

    n = len(X)

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

 

    HS = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(HS):

        print(ujd, wjd)

        HS = 0

 

    return HS
# Hopkins score

Hopkins_score=round(hopkins(mall),2)
print("{} is a good Hopkins score for Clustering.".format(Hopkins_score))
# Standarisation technique for scaling

scaler = StandardScaler()

mall_scaled = scaler.fit_transform(mall)
mall_scaled
mall_df1 = pd.DataFrame(mall_scaled, columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

mall_df1.head()

# Single linkage

plt.figure(figsize = (20,10))

mergings = linkage(mall_df1, method='single',metric='euclidean')

dendrogram(mergings)

plt.show()
# Complete Linkage

plt.figure(figsize = (20,10))

mergings = linkage(mall_df1, method='complete',metric='euclidean')

dendrogram(mergings)

plt.show()
# 4 clusters

cluster_labels = cut_tree(mergings, n_clusters=4).reshape(-1, )

cluster_labels
# Assign the label



mall_d['Cluster_Id'] = cluster_labels

mall_d.head()
## Number of customers in each cluster

mall_d['Cluster_Id'].value_counts(ascending=True)
mall_d.columns
plt.figure(figsize = (20,15))

plt.subplot(3,1,1)

sns.scatterplot(x = 'Age', y = 'Annual Income (k$)',hue='Cluster_Id',data = mall_d,legend='full',palette="Set1")

plt.subplot(3,1,2)

sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)',hue='Cluster_Id', data = mall_d,legend='full',palette="Set1")

plt.subplot(3,1,3)

sns.scatterplot(x = 'Spending Score (1-100)', y = 'Age',hue='Cluster_Id',data= mall_d,legend='full',palette="Set1")

plt.show()
 #Violin plot on Original attributes to visualize the spread of the data



fig, axes = plt.subplots(1,3, figsize=(20,5))



sns.violinplot(x = 'Cluster_Id', y = 'Age', data = mall_d,ax=axes[0])

sns.violinplot(x = 'Cluster_Id', y = 'Annual Income (k$)', data = mall_d,ax=axes[1])

sns.violinplot(x = 'Cluster_Id', y = 'Spending Score (1-100)', data=mall_d,ax=axes[2])

plt.show()
mall_d.head()
mall_d[['Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_Id']].groupby('Cluster_Id').mean()
group_0= mall_d[mall_d['Cluster_Id']==0]

group_0.head()
fig, axes = plt.subplots(1,3, figsize=(20,5))



sns.violinplot(x = 'Gender', y = 'Age', data = group_0,ax=axes[0])

sns.violinplot(x = 'Gender', y = 'Annual Income (k$)', data = group_0,ax=axes[1])

sns.violinplot(x = 'Gender', y = 'Spending Score (1-100)', data=group_0,ax=axes[2])

plt.show()
group_1= mall_d[mall_d['Cluster_Id']==1]

group_1.head()
fig, axes = plt.subplots(1,3, figsize=(20,5))



sns.violinplot(x = 'Gender', y = 'Age', data = group_1,ax=axes[0])

sns.violinplot(x = 'Gender', y = 'Annual Income (k$)', data = group_1,ax=axes[1])

sns.violinplot(x = 'Gender', y = 'Spending Score (1-100)', data=group_1,ax=axes[2])

plt.show()
group_2= mall_d[mall_d['Cluster_Id']==2]

group_2.head()
fig, axes = plt.subplots(1,3, figsize=(20,5))



sns.violinplot(x = 'Gender', y = 'Age', data = group_2,ax=axes[0])

sns.violinplot(x = 'Gender', y = 'Annual Income (k$)', data = group_2,ax=axes[1])

sns.violinplot(x = 'Gender', y = 'Spending Score (1-100)', data=group_2,ax=axes[2])

plt.show()
group_3= mall_d[mall_d['Cluster_Id']==3]

group_3.head()
fig, axes = plt.subplots(1,3, figsize=(20,5))



sns.violinplot(x = 'Gender', y = 'Age', data = group_3,ax=axes[0])

sns.violinplot(x = 'Gender', y = 'Annual Income (k$)', data = group_3,ax=axes[1])

sns.violinplot(x = 'Gender', y = 'Spending Score (1-100)', data=group_3,ax=axes[2])

plt.show()
mall_d[['Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_Id']].groupby('Cluster_Id').mean()