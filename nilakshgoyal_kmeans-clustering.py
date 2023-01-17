##importing libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.cluster import KMeans
##loading dataset
raw_data = pd.read_csv('../input/pca-kmeans-hierarchical-clustering/Country-data.csv')
raw_data
##in this we are considering only imports and exports of countries
data = raw_data.copy()
data = data.iloc[:,1:5]
data = data.drop(['child_mort','health'],axis=1)
data
##using kmeans clustering
kmeans = KMeans(3)
kmeans.fit(data)
known_clusters = kmeans.fit_predict(data)
known_clusters
##adding clusters to the table
clustered_data = data.copy()
clustered_data['clusters'] = known_clusters
clustered_data
plt.scatter(clustered_data['exports'],clustered_data['imports'],c=clustered_data['clusters'],cmap='rainbow')
plt.show()
