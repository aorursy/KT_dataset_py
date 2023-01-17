# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import os
# Importing the dataset

print(os.listdir("../input"))

# Importing the dataset

df = pd.read_csv('../input/Mall_Customers.csv')

df.head()
print ("The Shape of our dataset is: " + str(df.shape))
# Create new dataframe of annual income and spending score

X_spend = df[['Annual Income (k$)','Spending Score (1-100)']]

X_spend.head()
# Using the dendrogram to find the optimal number of clusters

import scipy.cluster.hierarchy as hcd

dendrogram = hcd.dendrogram(hcd.linkage(X_spend, metric='euclidean', method = 'ward'))

plt.title('Dendrogram', size=20)

plt.xlabel('Customers', size=15)

plt.ylabel('Euclidean Distances', size=15)

plt.show()
# Fitting Hierarchical Clustering with 3 Clusters to the dataset

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

X_spend['Cluster'] = hc.fit_predict(X_spend)
# Examine new dataframe with cluster column

X_spend.head()
# Define cluster colors

hc_colors = ['green' if c == 0 else 'blue' if c == 1 else 'purple' if c == 2 else 'black' if c == 3 else 'red' for c in X_spend.Cluster]



# Plot the scatter plot & clusters

fig = plt.figure(figsize=(10, 6))

plt.scatter(x="Annual Income (k$)",y="Spending Score (1-100)", data=X_spend, alpha=0.25, color = hc_colors)

plt.xlabel("Annual Income (k$)", size=15)

plt.ylabel("Spending Score (1-00)", size=15)

plt.title("Clusters of Spenders (3)", size=25)

plt.show()
# Fitting Hierarchical Clustering with 5 Clusters to the dataset

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

X_spend['Cluster'] = hc.fit_predict(X_spend)

# Define cluster colors

hc_colors = ['green' if c == 0 else 'blue' if c == 1 else 'purple' if c == 2 else 'black' if c == 3 else 'red' for c in X_spend.Cluster]



# Plot the scatter plot & clusters

fig = plt.figure(figsize=(10, 6))

plt.scatter(x="Annual Income (k$)",y="Spending Score (1-100)", data=X_spend, alpha=0.25, color = hc_colors)

plt.xlabel("Annual Income (k$)", size=15)

plt.ylabel("Spending Score (1-00)", size=15)

plt.title("Clusters of Spenders (5)", size=25)

plt.show()
df['cluster'] = X_spend['Cluster']

df[df['cluster']==3].describe()