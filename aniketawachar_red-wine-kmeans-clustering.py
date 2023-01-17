import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from scipy.stats import zscore

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df['quality'].value_counts()
df.head()
df_rows , df_cols = df.shape

print(df_rows)

print(df_cols)
df2 = df.loc[:, 'fixed acidity':'alcohol']
df2.describe().transpose()
import seaborn as sns

sns.pairplot(df2,diag_kind='kde')
df_scaled = df2.apply(zscore)

df_scaled.head()
model = KMeans(n_clusters = 6)
cluster_range = range( 1, 15 )

cluster_errors = []

for num_clusters in cluster_range:

  clusters = KMeans( num_clusters, n_init = 10 )

  clusters.fit(df_scaled)

  labels = clusters.labels_

  centroids = clusters.cluster_centers_

  cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:15]
# Elbow plot



plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
kmeans = KMeans(n_clusters=6, n_init = 15, random_state=2)
kmeans.fit(df_scaled)
centroids = kmeans.cluster_centers_
#Example to understand z-scale

a1=[7,5,4,3,8,36,42,28,30,35]

m1=np.mean(a1)

s1=np.std(a1)

a1z=(a1-m1)/s1

a1z

centroid_df = pd.DataFrame(centroids, columns = list(df_scaled) )
centroid_df
## creating a new dataframe only for labels and converting it into categorical variable

df_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))



df_labels['labels'] = df_labels['labels'].astype('category')
# Joining the label dataframe with the Wine data frame to create wine_df_labeled. Note: it could be appended to original dataframe

wine_df_labeled = df2.join(df_labels)
df_analysis = (wine_df_labeled.groupby(['labels'] , axis=0)).head(1599)  # the groupby creates a groupeddataframe that needs 

# to be converted back to dataframe. I am using .head(1599) for that

df_analysis
wine_df_labeled['labels'].value_counts()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=140)

kmeans.fit(df_scaled)

labels = kmeans.labels_

ax.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], df_scaled.iloc[:, 4],c=labels.astype(np.float), edgecolor='k')

ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('Acidity')

ax.set_ylabel('Sugar')

ax.set_zlabel('chlorides')

ax.set_title('3D plot of KMeans Clustering')