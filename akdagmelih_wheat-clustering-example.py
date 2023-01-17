# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(style="ticks", color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Seed_Data.csv')

data.sample(5)
data.info()
data.describe()
plt.figure(figsize=[8,8])

sns.heatmap(data.corr(), annot=True, cmap="YlGnBu")

plt.title('Correlations of the Features')

plt.show()
sns.countplot(data['target'], palette='husl')

plt.show()
i = sns.pairplot(data, vars = ['A', 'P', 'C', 'LK', 'WK', 'A_Coef', 'LKG'] ,hue='target', palette='husl')

plt.show()
a = sns.FacetGrid(data, col='target')

a.map(sns.boxplot, 'A', color='yellow', order=['0', '1', '2'])



p = sns.FacetGrid(data, col='target')

p.map(sns.boxplot, 'P', color='orange', order=['0', '1', '2'])



c = sns.FacetGrid(data, col='target')

c.map(sns.boxplot, 'C', color='red', order=['0', '1', '2'])



lk = sns.FacetGrid(data, col='target')

lk.map(sns.boxplot, 'LK', color='purple', order=['0', '1', '2'])



wk = sns.FacetGrid(data, col='target')

wk.map(sns.boxplot, 'WK', color='blue', order=['0', '1', '2'])



acoef = sns.FacetGrid(data, col='target')

acoef.map(sns.boxplot, 'A_Coef', color='cyan', order=['0', '1', '2'])



lkg = sns.FacetGrid(data, col='target')

lkg.map(sns.boxplot, 'LKG', color='green', order=['0', '1', '2'])
# Excluding target feature and create a new dataset:

df = data.iloc[:,0:7]

df.head(3)
from sklearn.cluster import KMeans



wcss = []



for k in range(1,10):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(df)

    wcss.append(kmeans.inertia_)

    

# Visualization of k values:



plt.plot(range(1,10), wcss, color='red')

plt.title('Graph of k values and WCSS')

plt.xlabel('k values')

plt.ylabel('wcss values')

plt.show()
# Now we know our best k value is 3, I am creating a new kmeans model:

kmeans2 = KMeans(n_clusters=3)



# Training the model:

clusters = kmeans2.fit_predict(df)



# Adding a label feature with the predicted class values:

df_k = df.copy(deep=True)

df_k['label'] = clusters
fig, (ax1, ax2) = plt.subplots(1,2)



ax1 = plt.subplot(1,2,1)

plt.title('Original Classes')

sns.scatterplot(x='A', y='P', hue='target', style='target', data=data, ax=ax1)



ax2 = plt.subplot(1,2,2)

plt.title('Predicted Classes')

sns.scatterplot(x='A', y='P', hue='label', style='label', data=df_k, ax=ax2)

plt.show()
print('Original Data Classes:')

print(data.target.value_counts())

print('-' * 30)

print('Predicted Data Classes:')

print(df_k.label.value_counts())
from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=[10,10])

merg = linkage(df, method='ward')

dendrogram(merg, leaf_rotation=90)

plt.title('Dendrogram')

plt.xlabel('Data Points')

plt.ylabel('Euclidean Distances')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hie_clus = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

cluster2 = hie_clus.fit_predict(df)



df_h = df.copy(deep=True)

df_h['label'] = cluster2
plt.title('Original Classes')

sns.scatterplot(x='A', y='P', hue='target', style='target', data=data)

plt.show()

plt.title('K-Means Classes')

sns.scatterplot(x='A', y='P', hue='label', style='label', data=df_k)

plt.show()

plt.title('Hierarchical Classes')

sns.scatterplot(x='A', y='P', hue='label', style='label', data=df_h)

plt.show()
print('Original Data Classes:')

print(data.target.value_counts())

print('-' * 30)

print('K-Means Predicted Data Classes:')

print(df_k.label.value_counts())

print('-' * 30)

print('Hierarchical Predicted Data Classes:')

print(df_h.label.value_counts())