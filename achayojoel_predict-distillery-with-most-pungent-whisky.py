# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

df=pd.read_csv('/kaggle/input/scotch-whisky-dataset/whisky.csv')

df.head()
df.shape
df.describe()
df.dtypes
corm=df.corr()

sns.heatmap(corm)
from sklearn.cluster import KMeans
SS = []

NC = range(1,20)

for k in NC:

    km = KMeans(n_clusters=k)

    km = km.fit(df.iloc[:,2:14])

    SS.append(km.inertia_)

plt.plot(NC,SS)

plt.xlabel('k')

plt.ylabel('SS')

plt.show()
X = df.iloc[:,2:14]

y = df['Distillery']
kmc5 = KMeans(n_clusters=5, random_state=0) # choose any arguments for the KMeans algorithm

cluster=kmc5.fit(X)

cluster
# to see the cluster centers 



ccenter5 = kmc5.cluster_centers_

ccenter5
from sklearn.manifold import MDS



plt.figure(figsize=(20, 20), dpi=50)

plt.rcParams["font.size"] = 15

labels=y



model = MDS(n_components=2, dissimilarity="euclidean", random_state=0)

out = model.fit_transform(X)

plt.scatter(out[:, 0], out[:, 1], c=kmc5.labels_,s=170)



plt.show()
# Add cluster ctreated ny kMeans to the dataframe

results=kmc5.labels_ 



df['Cluster']=results

## Distillery by cluster



t= df.groupby('Cluster')['Distillery'].unique()

t
# Create a column cluster_name, print the clusters and count  the distillery in each cluster



df['cluster_name']= df.groupby('Cluster')['Distillery'].transform('unique')

df['cluster_name'].value_counts()
## Cluster 0 distillery



df.query('Distillery=="AnCnoc"or Distillery== "Aultmore"')
## Cluster 1 Distillery



df.query('Distillery=="Ardbeg"or Distillery== "Clynelish"')
## Cluster 2 Distillery



df.query('Distillery=="Aberfeldy"or Distillery== "Aberlour"')
## Cluster 3 Distillery



df.query('Distillery=="Balmenach"or Distillery== "Dailuaine"')
## Cluster 4 Distillery



df.query('Distillery=="Ardmore"or Distillery== "Bowmore"')