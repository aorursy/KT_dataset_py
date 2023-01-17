# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_path = '../input/credit-card-data.csv' # Path to data file
data = pd.read_csv(data_path) 
data.head(15)
# 1. Provide three(3) plots of the data to assist in describing the initial data set
for col in data.columns[2:]:
    data[col].plot(kind='hist')
    plt.title('Histogram for '+col)
    plt.show()
for col in data.columns[2:]:
    data[col].plot(kind='density')
    plt.title('Density Plot for '+col)
    plt.show()
for col in data.columns[2:]:
    data[col].plot(kind='bar')
    plt.title('Bar Chart for '+col)
    plt.show()
# 2
cluster_data = data[['purchases','payments']]
cluster_data.head()

cluster_data.plot(kind='scatter',x='purchases',y='payments')
# Is there any missing data
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)

#retrieve just the values for all columns except customer id
data_values = cluster_data.iloc[ :, :].values
data_values

# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)
wcss = []
from sklearn.cluster import KMeans
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( data_values )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300) 
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data

cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
plt.show()

cluster_data['cluster'].value_counts().plot(kind='line',title='Distribution of Customers across groups')
plt.show()

sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data
grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True,)
kmeans.predict(pd.DataFrame({'purchases':[300],'payments':[40000]}))
kmeans.predict([[3,2]])
kmeans.algorithm
kmeans.cluster_centers_
kmeans
#3
cluster_data = data[['balance','credit_limit']]
cluster_data.head()

cluster_data.plot(kind='scatter',x='balance',y='credit_limit')
# Is there any missing data
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)


# perform imputation with median values
# Required because we have 1 missing for credi_limit
cluster_data = cluster_data.fillna( data.median() )

#retrieve just the values for all columns except customer id
data_values = cluster_data.iloc[ :, :].values
data_values

# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)
wcss = []
from sklearn.cluster import KMeans
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( data_values )
    wcss.append( kmeans.inertia_ )
    	
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300) 
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data

cluster_data['cluster'].value_counts().plot(kind='barh',title='Distribution of Customers across groups')
plt.show()


cluster_data['cluster'].value_counts().plot(kind='line',title='Distribution of Customers across groups')
plt.show()

sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data
grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True,)
kmeans.predict(pd.DataFrame({'balance':[300],'credit_limits':[40000]}))
kmeans.predict([[3,2]])
kmeans.algorithm
kmeans.cluster_centers_
kmeans