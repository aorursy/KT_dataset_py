# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_path = '../input/hr_data.csv' # Path to data file

data = pd.read_csv(data_path) 

data.head(15)
data.columns
data.describe()
data['salary'].value_counts().plot(kind='bar')
data['time_spend_company'].value_counts().plot(kind='pie')
for col in data.columns[2:8]:

    data[col].plot(kind='box')

    plt.title('Box Plot for '+col)

    plt.show()
cluster_data = data[['satisfaction_level','time_spend_company']]

cluster_data.head()
cluster_data.plot(kind='scatter',x='satisfaction_level',y='time_spend_company')
#Checking missing data

missing_data_results = cluster_data.isnull().sum()

print(missing_data_results)
data_values = cluster_data.iloc[ :, :].values

data_values
from sklearn.cluster import KMeans
#WCSS (within-cluster sums of squares) to find a number of clusters

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
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300) 

cluster_data["cluster"] = kmeans.fit_predict( data_values )

cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Employees across groups')
sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')

grouped_cluster_data
grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True)