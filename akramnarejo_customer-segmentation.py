# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

%matplotlib inline

plt.rcParams['figure.figsize'] = (15, 6)
# importing dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df
# lets get some info about the dataset

df.info()
# so lets rename the last 2 columns

df.rename(columns={'Annual Income (k$)':'Income', 'Spending Score (1-100)':'Spending_score'}, inplace=True)

df.head()
# let's remove the customerID column becuase it's useless

df.drop(columns=['CustomerID'], axis=1, inplace=True)
# lets check the data statisticuly or summarize the data

df.describe()
# lets visualize the features and relation with eachother

sns.pairplot(df)
X = df[['Income','Spending_score']]

dist_points_from_centroids = []

slscore = []

k = range(2,10)

for clusters in k:

    model = KMeans(n_clusters=clusters, max_iter=1000, random_state=10).fit(X)

    dist_points_from_centroids.append(model.inertia_)

    slscore.append(silhouette_score(X,model.labels_))

plt.xlabel("K")

plt.ylabel("inertia")

plt.title("Elbow Method")

plt.plot(k,dist_points_from_centroids)
plt.xlabel("K")

plt.ylabel("score")

plt.title("Silhouette score")

plt.plot(k, slscore)
# lets create the model 

kmeans = KMeans(n_clusters=5, max_iter=1000, random_state=10).fit(X)
# lets see that labels assigned to the clusters

kmeans.labels_
# lets make a new column named as cluster and assign labels into it.

df['cluster']=kmeans.labels_
df.head()
# lets see the number of poeple lie in each group

plt.title("clusters with the number of customers")

plt.xlabel("clusters")

plt.ylabel("Count")

df.cluster.value_counts().plot(kind='bar')
df.groupby(df.cluster).mean().plot(kind='bar')

plt.show()
plt.title("Men VS Women ratio in each cluster")

plt.ylabel("Count")

sns.countplot(x=df.cluster, hue=df.Gender)

plt.show()
plt.figure(figsize=(15,9))

g=sns.scatterplot(x='Income', y='Spending_score', hue='cluster', data=df,palette=['green','orange','brown','dodgerblue','red'], legend='full')