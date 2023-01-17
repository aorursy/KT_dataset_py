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

%matplotlib inline
data_path = '../input/credit-card-data.csv' #path to data file

data = pd.read_csv(data_path)

data.head(15)
data.describe()

data['tenure'].value_counts().plot(kind='bar')
data['tenure'].value_counts().sort_index().drop(12).plot(kind='bar')
for col in data.columns[2:]:

    data[col].plot(kind = 'box')

    plt.title('Box plot for'+col)

    plt.show()
cluster_data= data[['purchases','payments']]

cluster_data.head()
cluster_data.plot(kind='scatter', x= 'purchases',y='payments')
#is there any missing data

missing_data_results = cluster_data.isnull().sum()

print(missing_data_results)
data_values = cluster_data.iloc[:,:].values

data_values
from sklearn.cluster import KMeans

#use the elbow method

wcss=[]

for i in range(1,15):

    kmeans = KMeans(n_clusters=i,init="k-means++",n_init=10, max_iter=300)

    kmeans.fit_predict(data_values)

    wcss.append(kmeans.inertia_)

    

    plt.plot(wcss,'go-',label="WCSS")

    plt.title("Computing WCSS for KMeans++")

    plt.xlabel("Number of clusters")

    plt.ylabel("WCSS")

    plt.show()
kmeans.fit(data_values)

number_of_clusters = 3

kmeans = KMeans(n_clusters=number_of_clusters, init="k-means++",n_init=10, max_iter=300)
cluster_data['cluster']=kmeans.fit_predict(data_values)

cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title="Distribution of persons across Groups")

group_counts = cluster_data['cluster'].value_counts()

group_counts.name = 'Amount of persons in each group'

pd.DataFrame(group_counts)
import seaborn as sns
sns.pairplot(cluster_data,hue='cluster')
grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data.describe()

grouped_cluster_data.plot(subplots="True")
data.columns
len(data.columns)
co

        