

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
data_path='../input/hr_data.csv'

data=pd.read_csv(data_path)
data.head(10)
data.plot(kind="scatter", # or `us_gdp.plot.scatter(`

    x='number_project',

    y='average_montly_hours',

    title="Number of projects Montly Hours",

    figsize=(12,8)

)

plt.title("From %d to %d" % (

    data['number_project'].min(),

    data['number_project'].max()

),size=8)

plt.suptitle("Number Of Project",size=12)

plt.ylabel("Average Monthly Hours")

data.plot(kind="line", # or `us_gdp.plot.line(`

    x='number_project',

    y='satisfaction_level',

    title="Satisfaction Level",

    figsize=(12,8)

)

plt.title("From %d to %d" % (

    data['number_project'].min(),

    data['number_project'].max()

),size=8)

plt.suptitle("Number Of Project",size=12)

plt.ylabel("Satisfaction Level")
data.plot(kind='bar')

plt.title("From %d to %d" % (

    data['number_project'].min(),

    data['number_project'].max()

),size=8)

plt.suptitle(" Project Number%",size=12)
#import KMeans algorithm

from sklearn.cluster import KMeans
# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums wcss = []

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
kmeans.fit(data_values)

cluster_data['cluster']=kmeans.labels_

cluster_data.head(5)
cluster_data['cluster'].value_counts().plot(kind='bar',title="distribution of person across groups")

group_counts= cluster_data['cluster'].value_counts()

group_counts.name='Amount of person in each group'

pd.DataFrame(group_counts)
number_of_clusters=3

kmeans=KMeans(n_clusters=number_of_clusters,init="k-means++",n_init=10,max_iter=300)
cluster_data['cluster']=kmeans.fit_predict(data_values)

cluster_data
import seaborn as sns

sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')

grouped_cluster_data
grouped_cluster_data.plot(subplots=True,)