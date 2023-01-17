# Importing the libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
# Load the iris dataset

df = datasets.load_iris()

df = pd.DataFrame(df.data, columns = df.feature_names)

df.head() # See the first 5 rows
# To know number of rows and collumns

df.shape
# To find if any null value is present

df.isnull().sum()
# To see summary statistics

df.describe().T
# To find outliers

cols = df.columns

for i in cols:

    sns.boxplot(y=df[i])

    plt.show()
# To remove outliers from 'sepal width (cm)'

q1 = df['sepal width (cm)'].quantile(0.25)

q3 = df['sepal width (cm)'].quantile(0.75)

iqr = q3 - q1

df = df[(df['sepal width (cm)'] >= q1-1.5*iqr) & (df['sepal width (cm)'] <= q3+1.5*iqr)]

df.shape # To find out the number of rows and column after outlier treatment
# Blocplot for sepal width (cm) after outlier treatment

sns.boxplot(y=df['sepal width (cm)'])

plt.show()
# Standardizing to avoid bias

standard_scaler = StandardScaler()

df_norm = standard_scaler.fit_transform(df)
#To find the optimal no. of cluster

cluster_range = range(1,20)

cluster_errors = []



for num_cluster in cluster_range:

    clusters = KMeans(num_cluster, n_init = 10)

    clusters.fit(df_norm)

    labels = clusters.labels_

    centroids = clusters.cluster_centers_

    cluster_errors.append(clusters.inertia_)

    

clusters_df = pd.DataFrame({'num_cluster': cluster_range, 'cluster_errors': cluster_errors})

clusters_df[0:20]
#Ploting elbow curve or sree to find the no. of cluster

plt.figure(figsize=(12,6))

plt.plot(clusters_df.num_cluster, clusters_df.cluster_errors, marker = 'o') 

plt.xlabel('Values of K') 

plt.ylabel('Error') 

plt.title('The Elbow Method using Distortion') 

plt.show() 
# Creating object of the model and fitting it

model = KMeans(n_clusters=3, max_iter=50)

model.fit(df)
#analysis of cluster found

df.index = pd.RangeIndex(len(df.index))

df_km = pd.concat([df, pd.Series(model.labels_)], axis=1)

df_km.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'ClusterID']



km_clusters_Slength = pd.DataFrame(df_km.groupby(['ClusterID']).agg({'sepal length (cm)':'mean'}))

km_clusters_Swidth = pd.DataFrame(df_km.groupby(['ClusterID']).agg({'sepal width (cm)':'mean'}))

km_clusters_Plength = pd.DataFrame(df_km.groupby(['ClusterID']).agg({'petal length (cm)':'mean'}))

km_clusters_Pwidth = pd.DataFrame(df_km.groupby(['ClusterID']).agg({'petal width (cm)':'mean'}))
df2 = pd.concat([pd.Series([0,1,2]), km_clusters_Slength, km_clusters_Swidth, km_clusters_Plength, km_clusters_Pwidth

                ], axis=1)

df2.columns = ['ClusterID','sepal length (cm)_mean','sepal width (cm)_mean','petal length (cm)_mean',

               'petal width (cm)_mean']

df2.head()
sns.countplot(x=df_km.ClusterID)

plt.title('Count plot for ClusterID column')

plt.show()
#heirarchical clustering with full dendrogram

plt.figure(figsize=(15,7))

mergings = linkage(df_km, method = 'ward', metric='euclidean')



# set cut-off to 50

max_d = 7.08                # max_d as in max_distance

dendrogram(mergings,

           truncate_mode='lastp',  # show only the last p merged clusters

           p=150,                  # Try changing values of p

           leaf_rotation=90.,      # rotates the x axis labels

           leaf_font_size=8.,      # font size for the x axis labels

          )



plt.axhline(y=max_d, c='k')

plt.show()
#heirarchical clustering with full dendrogram for 50

plt.figure(figsize=(15,7))

mergings = linkage(df_km, method = 'ward', metric='euclidean')



# set cut-off to 50

max_d = 7.08                # max_d as in max_distance

dendrogram(mergings,

           truncate_mode='lastp',  # show only the last p merged clusters

           p=50,                  # Try changing values of p

           leaf_rotation=90.,      # rotates the x axis labels

           leaf_font_size=8.,      # font size for the x axis labels

          )



plt.axhline(y=max_d, c='k')

plt.show()



# Scatter plot to visualize the clusters

plt.figure(figsize=(10,7))

sns.scatterplot(x='sepal length (cm)',y='sepal width (cm)', data=df_km, hue='ClusterID', palette=['green','blue','red'])



# Plotting the centroids of the clusters

plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], 

            s = 100, c = 'yellow', label = 'Centroids')

plt.show()