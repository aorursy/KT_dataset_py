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
#Problem statement

#To determine the countries that are in the direst need of aid using socio-economic and health factors.



#Importing necessary libraries



import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
#Reading the CSV file



df = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')



#Reading the first five low of the dataframe



df.head()
#Reading the last five low of the dataframe



df.tail()
#Looking for the shape of the dataframe



print(df.shape)
# Looking for some more information like datatypes



df.info()
#Cleaning the data



#Looking for any missing values



round(100*(df.isnull().sum())/len(df.index),2)
#EDA analysis



df.describe()
#Here 'exports','health' and 'imports' are given as percentage of of GDP per capital sp converting these varaibles into their actual values.



df.exports = (df.exports*df.gdpp)/100

df.health = (df.health*df.gdpp)/100

df.imports = (df.imports*df.gdpp)/100



df.head()
features = df.columns[1:]

features
plt.figure(figsize=(15,15))

for i in enumerate(features):

    plt.subplot(3,3, i[0]+1)

    sns.distplot(df[i[1]])
plt.figure(figsize=(15,15))

for i in enumerate(features):

    plt.subplot(3,3, i[0]+1)

    sns.boxplot(df[i[1]])
plt.scatter(df['inflation'], df['child_mort'])

plt.show()
plt.scatter(df['total_fer'], df['life_expec'])

plt.show()
plt.scatter(df['inflation'], df['gdpp'])

plt.show()
g4 = df['gdpp'].quantile(0.85)



df['gdpp'][df['gdpp']>= g4] = g4





i4 = df['income'].quantile(0.95)



df['income'][df['income']>= i4] = i4



i3 = df['total_fer'].quantile(0.99)



df['total_fer'][df['total_fer']>= i3] = i3



e3 = df['exports'].quantile(0.89)



df['exports'][df['exports']>= e3] = e3



h3 = df['health'].quantile(0.86)



df['health'][df['health']>= h3] = h3



i2 = df['imports'].quantile(0.92)



df['imports'][df['imports']>= i2] = i2





i1 = df['inflation'].quantile(0.96)



df['inflation'][df['inflation']>= i1] = i1
plt.figure(figsize=(15,15))

for i in enumerate(features):

    plt.subplot(3,3, i[0]+1)

    sns.boxplot(df[i[1]])
df.describe()
#Clustering



# Check the hopkins



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



hopkins(df.drop('country', axis = 1))
#Scaling



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df1 = scaler.fit_transform(df.drop('country', axis = 1))

df1
df1 = pd.DataFrame(df1, columns = df.columns[1:])

df1.head()
#K-mean clustering



# Choose the value of K

# Silhouette score

# Elbow curve-ssd



from sklearn.metrics import silhouette_score

ss = []

for k in range(2, 11):

    kmean = KMeans(n_clusters = k).fit(df1)

    ss.append([k, silhouette_score(df1, kmean.labels_)])

temp = pd.DataFrame(ss)    

plt.plot(temp[0], temp[1])
ssd = []

for k in range(2, 11):

    kmean = KMeans(n_clusters = k).fit(df1)

    ssd.append([k, kmean.inertia_])

    

temp = pd.DataFrame(ssd)

plt.plot(temp[0], temp[1])
# K=3

# Final Kmean Clustering



kmean = KMeans(n_clusters =3 , random_state = 50)

kmean.fit(df1)
df_kmean = df.copy()
label  = pd.DataFrame(kmean.labels_, columns= ['label'])

label.head()
df_kmean = pd.concat([df_kmean, label], axis =1)

df_kmean.head()
df_kmean.label.value_counts()
# Plot the cluster

sns.scatterplot(x = 'child_mort', y = 'gdpp', hue = 'label', data = df_kmean, palette = 'Set1')
sns.scatterplot(x = 'income', y = 'gdpp', hue = 'label', data = df_kmean, palette = 'Set1')
sns.scatterplot(x = 'income', y = 'child_mort', hue = 'label', data = df_kmean, palette = 'Set1')
# Making sense out of the clsuters



df_kmean.drop(['country','exports','health', 'imports', 'inflation',

       'life_expec', 'total_fer'], axis = 1).groupby('label').mean().plot(kind = 'bar')
#plotting boxplot for 'label' and 'gdpp' 



sns.boxplot(x='label', y='gdpp', data=df_kmean)
#plotting boxplot for 'label' and 'income' 



sns.boxplot(x='label', y='income', data=df_kmean)
#plotting boxplot for 'label' and 'child_mort' 



sns.boxplot(x='label', y='child_mort', data=df_kmean)
# Observing Low income, Low GDP and High Child_mort from the boxplots and filter the data for that cluster.



df_kmean[df_kmean['label'] ==2]
#Flitering the required cluster with high child_mort, low gdpp and low income.



df_kmean[df_kmean['label'] == 2].sort_values(by = ['child_mort', 'gdpp','income'], ascending = [False, True,True]).head(10)
#Hierarchical Clustering



#Reading the first five rows of the dataframe



df.head()
#Plotting dendrogram using 'single' linkage method.



mergings = linkage(df1, method='single',metric ='euclidean')

dendrogram(mergings)

plt.show()
#Plotting dendrogram using 'complete' linkage method.



mergings = linkage(df1, method='complete',metric='euclidean')

dendrogram(mergings)

plt.show()
# 3 clusters



cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )

cluster_labels
# assign cluster labels



df1['cluster_labels'] = cluster_labels

df1.head()
#Concating with the actual dataframn with cluster labels and reading the firt five rows.



df_h = pd.concat([df,df1.cluster_labels],axis=1)



df_h.head()
#plotting boxplot for 'cluster_labels' and 'gdpp' 



sns.boxplot(x='cluster_labels', y='gdpp', data=df_h)
#plotting boxplot for 'cluster_labels' and 'income' 



sns.boxplot(x='cluster_labels', y='income', data=df_h)
#plotting boxplot for 'cluster_labels' and 'child_mort' 



sns.boxplot(x='cluster_labels', y='child_mort', data=df_h)
# By observing Low income, Low GDP and High Child_mort from the boxplot filter the data for that cluster.



df_h[df_h['cluster_labels'] ==0]
#Filter the cluster using high child_mort, low gdpp and low income.

#Listing out top 10 countries which are in direst need of aid.



df_h[df_h['cluster_labels'] == 0].sort_values(by = ['child_mort', 'gdpp','income'], ascending = [False, True,True]).head(10)