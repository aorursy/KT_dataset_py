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
# importing libraries



# Importing libraries and magic functions



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'

%matplotlib inline
# read data



df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
# check types

df.dtypes



# summary statistics on numeric features

df.describe()



# check for null values 

df.isna().sum()
# check for duplicates



df_duplicates = df[df.duplicated()]

df_duplicates
# check distribution



#histogram of whole ds

fig = plt.figure(figsize = (10,10))

ax = fig.gca()

#plt.axis('off')

df.hist(ax = ax)
# check for outliers

f, axes = plt.subplots(1, 3,figsize=(15,15))

sns.boxplot(df['Annual Income (k$)'],  orient='v' , ax=axes[0])

sns.boxplot(df.Age,  orient='v' , ax=axes[1])

sns.boxplot(df['Spending Score (1-100)'],  orient='v' , ax=axes[2])
# check outlier in annual income



outl = df[df['Annual Income (k$)'] > 130]

outl
# pairplot



sns.pairplot(df)
# check correlation



# correlation plot

corr = df.corr()

#plt.figure(figsize = (12,8))

sns.heatmap(corr, cmap = 'Wistia', annot= True)#, linewidths=.5)
# dropping CustID



df_dropped = df.drop(['CustomerID'], axis=1)

df_dropped.head()
# create dummies



df_dropped = pd.get_dummies(df_dropped, prefix=['Gender'])
# correlation check



df_dropped_corr = df_dropped.corr()



sns.heatmap(df_dropped_corr, cmap="YlGnBu")
# feature scaling

from sklearn.preprocessing import StandardScaler

# subtracts mean and then divides by standard deviation



# Your code here:

scaler = StandardScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df_dropped), columns = df_dropped.columns)

df_scaled.head()

df_scaled.describe()

df_dropped.describe()
# Clustering

# Load library

from sklearn.cluster import KMeans



# Create k-mean object

cluster = KMeans(n_clusters=7, random_state=0, n_jobs=-1)



# Train model

model = cluster.fit(df_scaled)



# adding cluster labels - Show cluster membership

df_scaled['labels'] = model.labels_
df_scaled.head()

df_scaled['labels'].value_counts()
# visualize clusters



# Your code here:

from matplotlib import pyplot

from mpl_toolkits.mplot3d import Axes3D

import random



fig = pyplot.figure()

ax = Axes3D(fig)



x = df_scaled.Age

y = df_scaled['Annual Income (k$)']

z = df_scaled['labels']



ax.scatter(x,y,z)



pyplot.show()
# Elbow & Silhouette

from yellowbrick.cluster import KElbowVisualizer



model = KMeans()

visualizer = KElbowVisualizer(model, k=(3,12), metric ='silhouette')

visualizer.fit(df_scaled)

visualizer.poof()
# Elbow metric distortion

from yellowbrick.cluster import KElbowVisualizer



model = KMeans()

visualizer = KElbowVisualizer(model, k=(3,17))

visualizer.fit(df_scaled)

visualizer.poof()
# Elbow metric 'calinski_harabasz'/hierarchical

model = KMeans()

visualizer = KElbowVisualizer(model, k=(3,17), metric ='calinski_harabasz')

visualizer.fit(df_scaled)

visualizer.poof()
# Silhouette Score

from sklearn.metrics import silhouette_score



preds = cluster.fit_predict(df_scaled)

centers = cluster.cluster_centers_



score = silhouette_score (df_scaled, preds, metric='euclidean')

score
# DBSCAN Clustering



from sklearn.cluster import DBSCAN



# Create meanshift object

cluster2 = DBSCAN(eps=0.7, min_samples=6)



# Train model

model2 = cluster2.fit(df_scaled)



# Show cluster membership

df_scaled['labels_DBSCAN'] = model2.labels_



# value counts

df_scaled['labels_DBSCAN'].value_counts()
# Silhouette Score

from sklearn.metrics import silhouette_score



score2 = silhouette_score (df_scaled, df_scaled['labels_DBSCAN'])

score2
# Hierarchical Clustering



from sklearn.cluster import AgglomerativeClustering



# Create meanshift object

cluster3 = AgglomerativeClustering(n_clusters=10)



# Train model

model3 = cluster3.fit(df_scaled)



# Show cluster membership

df_scaled['labels_agglo'] = model3.labels_

# Silhouette Score



score3 = silhouette_score (df_scaled, df_scaled['labels_agglo'])

score3
# Elbow metric 'calinski_harabasz'/hierarchical

visualizer = KElbowVisualizer(model3, k=(3,17), metric ='calinski_harabasz')

visualizer.fit(df_scaled)

visualizer.poof()
f, axes = plt.subplots(1,3, figsize=(15,5))



sns.scatterplot(  y="Age", x= "Spending Score (1-100)", hue='labels', data=df_scaled,  ax=axes[0])

sns.scatterplot(  y="Age", x= "Spending Score (1-100)", hue='labels_DBSCAN', data=df_scaled, ax=axes[1])

sns.scatterplot(  y="Age", x= "Spending Score (1-100)", hue='labels_agglo', data=df_scaled, ax=axes[2])
