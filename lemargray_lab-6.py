# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_path = '../input/credit-card-data.csv' # Path to data file
data = pd.read_csv(data_path)
data.head(15)
data.columns
data.describe()
data['tenure'].value_counts().plot(kind='bar')
# Let's view the distribution of the data, where is it possible to find groups?
# We are using boxplots of all the columns except the first (cust_id which is a string)
for col in data.columns[2:]:
    data[col].plot(kind='box')
    plt.title('Box Plot for '+col)
    plt.show()
cluster_data = data[['purchases','payments']]
cluster_data.head()
cluster_data.plot(kind='scatter',x='purchases',y='payments')
# Is there any missing data
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

kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300)
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data
cluster_data['cluster'].value_counts()

sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data
grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True,)
# first plot. scatter plot of credit_limit vs purchases in the dataset
#This scatter plot shows that the more purchases a customer does the higher the cr
#for that customer.
data.plot(kind="scatter", x='credit_limit',y='purchases',color='green', title="Credit limits per purchase")
data.plot(kind="line", x='credit_limit',y='payments', color='brown',title="Credit Limit per Paymeny")
missing_data_results = cluster_data1.isnull().sum()
print(missing_data_results)
data_values = cluster_data1.iloc[:,1:].values
data_values
from sklearn.cluster import KMeans
#clusters recommended for use is 3
ks = range(1, 15)
inertias = []
for k in ks:
 model = KMeans(n_clusters=k)
 model.fit(cluster_data1)
 inertias.append(model.inertia_)
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')

plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()
