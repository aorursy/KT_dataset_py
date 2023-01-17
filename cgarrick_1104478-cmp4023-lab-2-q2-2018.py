# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_path = '../input/hr_data.csv' # Path to data file

data = pd.read_csv(data_path)

data.head(15)

# What columns are in the data set ? Do they have spaces that I should consider

data.columns
data.describe()
# Let's view the distribution of the data, where is it possible to find groups?

# We are using boxplots of all the columns except sales and salary 

for col in data.columns[0:8]:

    data[col].plot(kind='hist')

    plt.title('Histogram Plot for '+col)

    plt.show()
for col in data.columns[0:8]:

    data[col].plot(kind='Density')

    plt.title('Density Plot for '+col)

    plt.show()
for col in data.columns[0:8]:

    data[col].plot(kind='Box')

    plt.title('Box Plot for '+col)

    plt.show()
cluster_data = data[['number_project','average_montly_hours']]

cluster_data.head()


cluster_data.plot(kind='scatter',x='number_project',y='average_montly_hours')
# Is there any missing data

missing_data_results = cluster_data.isnull().sum()

print(missing_data_results)



# perform imputation with median values

# not require since none missing

#cluster_data = cluster_data.fillna( data.median() )
#import KMeans algorithm

from sklearn.cluster import KMeans
data_values = cluster_data.iloc[ :, :].values

data_values
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
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300) 

cluster_data["cluster"] = kmeans.fit_predict( data_values )

cluster_data
kmeans.fit(data_values)

cluster_data['cluster']=kmeans.labels_

cluster_data.head(5)
cluster_data['cluster'].value_counts().plot(kind='bar',title="distribution of person across groups")

group_counts= cluster_data['cluster'].value_counts()

group_counts.name='Amount of person in each group'

pd.DataFrame(group_counts)
import seaborn as sns

sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')

grouped_cluster_data
grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True,)