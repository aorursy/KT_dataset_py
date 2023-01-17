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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df.shape
df.isnull().sum()
df.head()
df.rename(index=str, columns={'Annual Income (k$)': 'Income',

                              'Spending Score (1-100)': 'Score'}, inplace=True)

df.head()
data=df.drop(["CustomerID"],axis=1)
sns.pairplot(data=data,hue="Gender",aspect=1.5)

plt.show()
data.drop("Gender",axis=1,inplace=True)
from sklearn.cluster import KMeans
clusters=[]

for i in range (1,11):

    km=KMeans(n_clusters=i,init="k-means++",random_state=123)

    km.fit(data)

    clusters.append(km.inertia_)



plt.plot(range(1,11),clusters)
# 3 cluster

km3 = KMeans(n_clusters=3).fit(data)



data['Labels'] = km3.labels_

plt.figure(figsize=(12, 8))

sns.scatterplot(data['Income'], data['Score'], hue=data['Labels'], 

                palette=sns.color_palette('hls', 3))

plt.title('KMeans with 3 Clusters')

plt.show()

data.head()
#with 5 clusters

km5=KMeans(n_clusters=5).fit(data)

data['Labels'] = km5.labels_

plt.figure(figsize=(12, 8))

sns.scatterplot(data['Income'], data['Score'], hue=data['Labels'], 

                palette=sns.color_palette('hls', 5))

plt.title('KMeans with 5 Clusters')

plt.show()
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

sns.swarmplot(x='Labels', y='Income', data=data, ax=ax)

ax.set_title('Labels According to Annual Income')



ax = fig.add_subplot(122)

sns.swarmplot(x='Labels', y='Score', data=data, ax=ax)

ax.set_title('Labels According to Scoring History')



plt.show()
from sklearn.cluster import AgglomerativeClustering 



agglom = AgglomerativeClustering(n_clusters=5, linkage='average').fit(data)



data['Labels'] = agglom.labels_

plt.figure(figsize=(12, 8))

sns.scatterplot(data['Income'], data['Score'], hue=data['Labels'], 

                palette=sns.color_palette('hls', 5))

plt.title('Agglomerative with 5 Clusters')

plt.show()

from scipy.cluster import hierarchy

from scipy.spatial import distance_matrix 



dist = distance_matrix(data,data)

print(dist)

Z = hierarchy.linkage(dist, 'complete')

plt.figure(figsize=(18, 50))

dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')