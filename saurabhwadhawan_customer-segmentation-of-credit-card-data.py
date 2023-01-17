# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')

df.head()
df.isnull().sum()
df['MINIMUM_PAYMENTS'].describe()
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
df['MINIMUM_PAYMENTS'].isnull().sum()
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean())
df = df.drop('CUST_ID', axis=1)
cluster_range = range(2,25)

cluster_errors = []

for i in cluster_range:

    clusters = KMeans(i) 

    clusters.fit(df) 

    labels = clusters.labels_

    centroids = clusters.cluster_centers_

    cluster_errors.append(clusters.inertia_)

clusters_df = pd.DataFrame({"num_clusters":cluster_range, "cluster_errors":cluster_errors})

clusters_df[1:20]

plt.figure(figsize=(15,10))

plt.plot(clusters_df.num_clusters,clusters_df.cluster_errors, marker = '*',color='violet')# Scree Plot/Elbow Curve

plt.xlabel("Number of Clusters")

plt.ylabel("Cluster Errors")

plt.grid(True)

plt.show()
kmeans = KMeans(n_clusters=7, n_init=15, random_state=123)

kmeans.fit(df)

df_labeled = pd.DataFrame(kmeans.labels_, columns = list(['labels']))

df_labeled['labels'] = df_labeled['labels'].astype('category')
plt.figure(figsize=(10,8))

df_labeled['labels'].value_counts().plot.bar(color='red')

plt.xlabel("Labels")

plt.ylabel("Count of Customers")

plt.title("Number of Customers in Each Category")

plt.show()
df = df.join(df_labeled)
zero = df[df['labels'] == 0].PURCHASES.mean()

one = df[df['labels'] == 1].PURCHASES.mean()

two = df[df['labels'] == 2].PURCHASES.mean()

three = df[df['labels'] == 3].PURCHASES.mean()

four = df[df['labels'] == 4].PURCHASES.mean()

five = df[df['labels'] == 5].PURCHASES.mean()

six = df[df['labels'] == 6].PURCHASES.mean()



indices = ['0','1','2', '3', '4', '5', '6']

bar = pd.DataFrame([zero, one, two, three, four, five, six], index = indices)

bar.plot.bar(color='green')

plt.xlabel('Label')

plt.ylabel('Mean Purchase Value')

plt.title("Mean Purchase Value of Each Category")

plt.show()
from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=(20,10))



merg = linkage(df.drop('labels',1), method='ward')

dendrogram(merg, leaf_rotation = 360)

plt.title('Dendrogram')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hier_clus = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')

cluster = hier_clus.fit_predict(df.drop('labels',1))



df['Agg_label'] = cluster
print("Agglomerative labels")

df['Agg_label'].value_counts()
print("Kmeans labels")

df['labels'].value_counts()
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,10))



ax1 = plt.subplot(1,2,1)

plt.title('KMeans Predicted Classes')

sns.scatterplot(x='PURCHASES', y='CASH_ADVANCE', style='labels', data=df,ax=ax1)



ax2 = plt.subplot(1,2,2)

plt.title('Hierarchical Predicted Classes')

sns.scatterplot(x='PURCHASES', y='CASH_ADVANCE', style = 'Agg_label', data=df, ax=ax2)



plt.show()