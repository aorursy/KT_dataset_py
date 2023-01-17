import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns



sns.set()

%matplotlib inline
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering
crime = pd.read_csv("../input/usarrests/USArrests.csv")
#peeking at the dataset



crime.head(5)
# Let's see how many rows and columns we got!



crime.shape
#Let's see some facts here



crime.info()
# Let's get some statistics summary



crime.describe()
crime.isnull().sum()
# Renaming the column as Unnmaed doesn't make sense.



crime = crime.rename(columns={'Unnamed: 0':'State'})
crime.head()
plt.figure(figsize=(20,5))

crime.groupby('State')['Murder'].max().plot(kind='bar')
plt.figure(figsize=(20,5))

crime.groupby('State')['Assault'].max().plot(kind='bar')
plt.figure(figsize=(20,5))

crime.groupby('State')['Rape'].max().plot(kind='bar')
plt.figure(figsize=(20,5))

crime.groupby('State')['UrbanPop'].max().plot(kind='bar')
plt.figure(figsize=(10,5))

plt.scatter('UrbanPop','Murder',data=crime)

plt.xlabel('Urban Population')

plt.ylabel('Murder Rate')
plt.figure(figsize=(10,5))

plt.scatter('UrbanPop','Rape',data=crime)

plt.xlabel('Urban Population')

plt.ylabel('Rape Rate')
plt.figure(figsize=(10,5))

plt.scatter('UrbanPop','Assault',data=crime)

plt.xlabel('Urban Population')

plt.ylabel('Assault Rate')
data = crime.iloc[:,1:].values
scaled_data = scaler.fit_transform(data)
plt.figure(figsize=(20,5))

plt.title("Crime Rate Dendograms")

dend = sch.dendrogram(sch.linkage(scaled_data, method='single'))

plt.xlabel('Crime Rate')

plt.ylabel('Euclidean distances')
plt.figure(figsize=(20,5))

plt.title("Crime Rate Dendograms")

dend = sch.dendrogram(sch.linkage(scaled_data, method='complete'))

plt.xlabel('Crime Rate')

plt.ylabel('Euclidean distances')
plt.figure(figsize=(20,5))

plt.title("Crime Rate Dendograms")

dend = sch.dendrogram(sch.linkage(scaled_data, method='average'))

plt.xlabel('Crime Rate')

plt.ylabel('Euclidean distances')
# With Ward method

plt.figure(figsize=(20,8))

dendrogram = sch.dendrogram(sch.linkage(data, method  = "ward"))

plt.title('Dendrogram')

plt.xlabel('Crime Rate')

plt.ylabel('Euclidean distances')

plt.show()
# Fit the Agglomerative Clustering

 

AC = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='ward')
# Fit and predict to have the cluster labels.

y_pred =AC.fit_predict(data)

y_pred
# Fetch the cluster labels

crime['cluster labels']= y_pred
# Let's see which State falls in which cluster

crime[['State','cluster labels']]
plt.figure(figsize=(10,5))

sns.boxplot(x='cluster labels', y='Murder', data=crime)
plt.figure(figsize=(10,5))

sns.boxplot(x='cluster labels', y='Rape', data=crime)
plt.figure(figsize=(10,5))

sns.boxplot(x='cluster labels', y='Assault', data=crime)
Safe_Zone= crime.groupby('cluster labels')['State'].unique()[0]

Safe_Zone
Danger_Zone= crime.groupby('cluster labels')['State'].unique()[1]

Danger_Zone
Moderate_Zone= crime.groupby('cluster labels')['State'].unique()[2]

Moderate_Zone
plt.figure(figsize=(10,5))

plt.scatter(data[y_pred==0, 0], data[y_pred==0, 1], s=100, c='red', label ='Safe_Zone')

plt.scatter(data[y_pred==1, 0], data[y_pred==1, 1], s=100, c='blue', label ='Danger_Zone')

plt.scatter(data[y_pred==2, 0], data[y_pred==2, 1], s=100, c='green', label ='Moderate_Zone')

plt.legend()

plt.show()