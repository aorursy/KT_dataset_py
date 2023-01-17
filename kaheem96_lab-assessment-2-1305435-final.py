# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

data_path = '../input/credit-card-data.csv' # Path to data file
data = pd.read_csv(data_path) 
data.head(15)
# What columns are in the data set ? Do they have spaces that I should consider
data.columns

data.describe()

for col in data.columns[2:]:
    data[col].plot(kind='density')
    plt.title('Density Plot for '+col)
    plt.show()
for col in data.columns[2:]:
    data[col].plot(kind='hist')
    plt.title('Histogram Plot for '+col)
    plt.show()
for col in data.columns[2:]:
    data[col].plot(kind='line')
    plt.title('Histogram Plot for '+col)
    plt.show()
cluster_data = data[['purchases','payments']]
cluster_data.head()
cluster_data.plot(kind='scatter',x='purchases',y='payments')
# Is there any missing data
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)

# perform imputation with median values
# not require since none missing
#cluster_data = cluster_data.fillna( data.median() )
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
cluster_data['cluster'].value_counts().plot(kind='pie',title='Numerical Portion each groups')
fig = plt.gcf()
fig.set_size_inches(7,6)
plt.show()
cluster_data['cluster'].value_counts().plot(kind='barh',title='Distribution of Customers across groups')
fig = plt.gcf()
fig.set_size_inches(7,4)
plt.show()
data.head(5)
cluster_data = data[['payments','credit_limit']]
cluster_data.head()
cluster_data.plot(kind='scatter',x='payments',y='credit_limit')
# Is there any missing data
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)


# perform imputation with median values
# Required because we have 1 missing for credi_limit


cluster_data = cluster_data.fillna( data.median() )
#check to see if data is still missing
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)


#retrieve just the values for all columns except customer id
data_values = cluster_data.iloc[ :, :].values
data_values
#import KMeans algorithm
from sklearn.cluster import KMeans

# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)
wcss = []
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( data_values )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300) 
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data

grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True,)
kmeans.predict(pd.DataFrame({'payments':[300],'credit_limit':[40000]}))
kmeans.predict([[3,2]])
kmeans.algorithm

kmeans.cluster_centers_

kmeans
