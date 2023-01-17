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
df =pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.drop(['id', 'name', 'host_id', 'host_name'], axis=1, inplace=True)
df.info()
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns



fig = plt.figure(figsize=(12,12))

sns.heatmap(df.isnull())
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

df['last_review'].fillna('NaN', inplace=True)
df.info()
df.nunique()
df.describe()
fig = plt.figure(figsize=(8,8))

sns.heatmap(df.corr(), annot=True, linewidth=0.5)
fig = plt.figure(figsize=(13,8))

sns.scatterplot(data=df, x='reviews_per_month', 

                y='number_of_reviews', 

                hue='neighbourhood_group',

                style='neighbourhood_group',

                s=50, alpha=0.5)

plt.xlim(0, 25)
cols=df.drop(['latitude', 'longitude', 'neighbourhood_group', 

              'neighbourhood', 'room_type', 'last_review', 

              'reviews_per_month'], axis=1)

f, axes = plt.subplots(2, 3, figsize=(15, 8))

axes=axes.ravel()

for i,col in enumerate(cols):

    plot=sns.boxplot(x=col, data=df, ax=axes[i], orient='v')
df[df['price'] == 0].count()
df = df[df['price'] != 0]
fig = plt.figure(figsize=(12,8))

sns.scatterplot(data=df, x='longitude', y='latitude', 

                hue='neighbourhood_group', style='neighbourhood_group',

                s=50, alpha=0.5)
df.info()
fig = plt.figure(figsize=(12,8))

sns.boxplot(data=df, x='price', 

            y='neighbourhood_group',

            hue='room_type')

plt.xlim(0, 750)
fig = plt.figure(figsize=(12,8))

sns.boxplot(data=df, x='minimum_nights', 

            y='neighbourhood_group',

            hue='room_type')

plt.xlim(0, 25)
fig = plt.figure(figsize=(12,8))

sns.boxplot(data=df, x='reviews_per_month', 

            y='neighbourhood_group',

            hue='room_type')

plt.xlim(0, 10)
df.info()
df_group = df.groupby('neighbourhood').agg({'number_of_reviews':'mean',

                                            'reviews_per_month':'mean', 

                                            'latitude': 'mean', 'longitude': 'mean', 

                                            'price': 'mean', 

                                            'minimum_nights': 'mean'})
df_group
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

df_scaled = scaler.fit_transform(df_group)
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

df_pca = pca.fit_transform(df_scaled)

pca.components_
pca_table = pd.DataFrame(pca.components_, columns=df_group.columns)

pca_table
df_pca
import scipy.cluster.hierarchy as sch



fig = plt.figure(figsize=(14,8))

dendrogram = sch.dendrogram(sch.linkage(df_pca, method='ward'))

plt.title('Dendrograms', size=30)

plt.xlabel('Neighbourhood', size=20)

plt.ylabel('Euclidean Distances', size=20)
fig = plt.figure(figsize=(14,8))

dendrogram = sch.dendrogram(sch.linkage(df_pca, method='ward'))

plt.axhline(y=2.5, color='y', linestyle='--')

plt.title('Dendrograms', size=30)

plt.xlabel('Neighbourhood', size=20)

plt.ylabel('Euclidean Distances', size=20)
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(df_pca)

y_hc
x1 = np.array(df_pca[y_hc==0])[:,0]

y1 = np.array(df_pca[y_hc==0])[:,1]

x2 = np.array(df_pca[y_hc==1])[:,0]

y2 = np.array(df_pca[y_hc==1])[:,1]

x3 = np.array(df_pca[y_hc==2])[:,0]

y3 = np.array(df_pca[y_hc==2])[:,1]
fig = plt.figure(figsize=(12,8))

sns.scatterplot(x1, y1, color='blue', label='Cluster1', s=70, alpha=0.7)

sns.scatterplot(x2, y2, color='red', label='Cluster2', s=70, alpha=0.7)

sns.scatterplot(x3, y3, color='green', label='Cluster3', s=70, alpha=0.7)

plt.title('Clusters of Neighbourhood', size=30)