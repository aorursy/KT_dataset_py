import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import datetime as dt

import sklearn

from sklearn.preprocessing import MinMaxScaler

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
# read the dataset

mall_df = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv", sep=",", encoding="ISO-8859-1", header=0)

mall_df.head(100)
#Check Null Values

mall_df.isnull().sum()
#Inspect the DataFrame

mall_df.info()
sns.violinplot(mall_df['Age']);
sns.violinplot(mall_df['Annual Income (k$)']);

sns.violinplot(mall_df['Spending Score (1-100)'])

plt.show()
#BIvariate Analysis

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

cluster_type = mall_df.groupby(['Gender'])['Annual Income (k$)'].mean().reset_index()

ax=sns.barplot(x = 'Gender', y='Annual Income (k$)', data=cluster_type)

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))



plt.subplot(1,2,2)

cluster_type = mall_df.groupby(['Gender'])['Spending Score (1-100)'].mean().reset_index()

ax=sns.barplot(x = 'Gender', y='Spending Score (1-100)', data=cluster_type)

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))



plt.show()
#Binning the Age of the Customers

bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]

slot = ['0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70+']



mall_df['Age_Range']=pd.cut(mall_df['Age'],bins,labels=slot)
#Which age People mostly visit the Mall?

plt.figure(figsize=(12,6))

edu=sns.countplot(x="Age_Range", data=mall_df)

edu.set_xticklabels(edu.get_xticklabels(),rotation=90)

plt.show()
#BIvariate Analysis

plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

cluster_type = mall_df.groupby(['Age_Range'])['Annual Income (k$)'].mean().reset_index()

sns.barplot(x = 'Age_Range', y='Annual Income (k$)', data=cluster_type)

plt.subplot(1,2,2)

cluster_type = mall_df.groupby(['Age_Range'])['Spending Score (1-100)'].mean().reset_index()

sns.barplot(x = 'Age_Range', y='Spending Score (1-100)', data=cluster_type)

plt.show()
mall_df.drop('Age_Range',axis=1,inplace=True)

mall_df.head()
from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) # heuristic from article [1]

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

hopkins(mall_df.drop(['CustomerID','Gender'],axis=1))
mall_df.head()
mall_df_data=mall_df.drop(['CustomerID','Gender','Age'],axis=1)

# instantiate

scaler = MinMaxScaler()



# fit_transform

mall_df_scaled = scaler.fit_transform(mall_df_data)

mall_df_scaled.shape
#Converting the scaled data to data frame

mall_df_scaled = pd.DataFrame(mall_df_scaled)

mall_df_scaled.columns = mall_df_data.columns

mall_df_scaled.head()
# single linkage

plt.figure(figsize=(30,15))

mergings = linkage(mall_df_scaled, method="single", metric='euclidean')

dendrogram(mergings)

plt.show()
# complete linkage

plt.figure(figsize=(30,15))

mergings = linkage(mall_df_scaled, method='complete', metric='euclidean')

dendrogram(mergings)

plt.show()
# 4 clusters

cluster_l = cut_tree(mergings, n_clusters=4).reshape(-1, )

cluster_l
# assign cluster labels

mall_df['cluster_id'] = cluster_l

mall_df.head()
plt.figure(figsize=(20,10))



sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', hue = 'cluster_id', data = mall_df, palette = 'Set1');
mall_df['cluster_id'].value_counts()
#Profiling the clusters

mall_df.drop(['CustomerID','Age','Gender'],axis=1).groupby('cluster_id').mean().plot(kind='bar')

plt.show()
from sklearn.cluster import DBSCAN

from collections import Counter
model=DBSCAN(eps=0.1,min_samples=10).fit(mall_df_scaled)

print(model.labels_)

# assign cluster labels

mall_df['cluster_lab'] = model.labels_

mall_df.head()
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']

vectorizer = np.vectorize(lambda : colors[x % len(colors)])
plt.scatter(mall_df['Annual Income (k$)'], mall_df['Spending Score (1-100)'], c=mall_df['cluster_lab']);
plt.figure(figsize=(20,10))



sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', hue = 'cluster_lab', data = mall_df, palette = 'Set1');
mall_df_scaled.describe()
#Profiling the clusters

mall_df.drop(['CustomerID','Age','Gender','cluster_id'],axis=1).groupby('cluster_lab').mean().plot(kind='bar')

plt.show()
#Valuable Customers

mall_df[mall_df['cluster_lab']==2]
#need to market on this cluster to improve the spending score

mall_df[mall_df['cluster_lab']==3]
#outliers

mall_df[mall_df['cluster_lab']==1]
#moderate performers

mall_df[mall_df['cluster_lab']==-1]