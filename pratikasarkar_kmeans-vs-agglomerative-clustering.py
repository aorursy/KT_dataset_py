import numpy as np

import pandas as pd

import seaborn as sns

from scipy.stats import zscore

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
sns.set(style = 'ticks',color_codes=True)
df2 = pd.read_csv('../input/matplotlib-datasets/iris_dataset.csv')
df2
df2.drop('species',axis = 1,inplace = True)
df_scaled = df2.apply(zscore)
sns.pairplot(df_scaled,diag_kind='kde')
clusters_range = range(1,15)

inertia = []

for num_clust in clusters_range:

  model = KMeans(n_clusters = num_clust,random_state = 2)

  model.fit(df_scaled)

  inertia.append(model.inertia_)
plt.plot(clusters_range,inertia,marker = 'o')
kmeans = KMeans(n_clusters=3,random_state=2)

kmeans.fit(df_scaled)

df2['class'] = kmeans.labels_

df2
df2['class'].value_counts()
sns.pairplot(df2,hue = 'class')
from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=3)

agc.fit(df_scaled)
df_scaled['agc_class'] = agc.labels_
df_scaled['agc_class'].value_counts()
df_scaled
grps = df_scaled.groupby('agc_class')
grp0 = grps.get_group(0)

grp1 = grps.get_group(1)

grp2 = grps.get_group(2)
c0 = np.array([grp0['petal_length'].mean(),grp0['petal_width'].mean(),grp0['sepal_length'].mean(),grp0['sepal_width'].mean()])

c1 = np.array([grp1['petal_length'].mean(),grp1['petal_width'].mean(),grp1['sepal_length'].mean(),grp1['sepal_width'].mean()])

c2 = np.array([grp2['petal_length'].mean(),grp2['petal_width'].mean(),grp2['sepal_length'].mean(),grp2['sepal_width'].mean()])
df_scaled.columns
inertia_0 = np.sum(((grp0['petal_length'] - c0[0])**2) + ((grp0['petal_width'] - c0[1])**2) + ((grp0['sepal_length'] - c0[2])**2) + ((grp0['sepal_width'] - c0[3])**2))

inertia_1 = np.sum(((grp1['petal_length'] - c1[0])**2) + ((grp1['petal_width'] - c1[1])**2) + ((grp1['sepal_length'] - c1[2])**2) + ((grp1['sepal_width'] - c1[3])**2))

inertia_2 = np.sum(((grp2['petal_length'] - c2[0])**2) + ((grp2['petal_width'] - c2[1])**2) + ((grp2['sepal_length'] - c2[2])**2) + ((grp2['sepal_width'] - c2[3])**2))

total_inertia = inertia_0 + inertia_1 + inertia_2

total_inertia
df_scaled.drop('agc_class',axis = 1,inplace = True)
from scipy.cluster.hierarchy import dendrogram,linkage

plt.figure(figsize=(10,5))

plt.xlabel('sample index')

plt.ylabel('distance')

z = linkage(df_scaled,method='ward')

dendrogram(z,leaf_rotation=90,p = 5,color_threshold=10,leaf_font_size=10,truncate_mode='level')

plt.tight_layout()