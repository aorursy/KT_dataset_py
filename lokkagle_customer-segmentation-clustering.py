# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt


# loading data

data = pd.read_csv('/kaggle/input/Mall_Customers.csv')

data.head()
data.info()


data.describe().T
data.isna().sum()
mall_data = data.copy(deep= True)

mall_data.head()
# drop unnecessary attributes

mall_data.drop(columns= 'CustomerID', inplace= True)

mall_data.head()
sns.countplot(x = 'Gender', data = mall_data)

plt.show()
sns.scatterplot(x = 'Age', y = 'Annual Income (k$)', data = mall_data, hue = 'Gender')

sns.jointplot(x = 'Age', y = 'Annual Income (k$)', data = mall_data)

plt.show()
sns.scatterplot(x = 'Age', y = 'Spending Score (1-100)', data = mall_data, hue = 'Gender')

sns.jointplot(x = 'Age', y = 'Spending Score (1-100)', data = mall_data)

plt.show()
sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', data = mall_data, hue = 'Gender')

sns.jointplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', data = mall_data)

plt.show()


## preparing the data

mall_data.iloc[:, [2,3]].head()


# check data distribution

plt.style.use('seaborn')

mall_data.iloc[:, [2,3]].hist(figsize = (14,3))

plt.show()
X = mall_data.iloc[:, [2,3]].values
# Finding optimal number of clusters using elbow method

from sklearn.cluster import KMeans

wcss = []



for i in range(1,11):

    kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state= 15)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
# visualization for optimal number of clusters

plt.plot(range(1,11), wcss, marker = 'o',linestyle = 'dashed')

plt.text(4.9,40000, 'X',bbox=dict(facecolor='red', alpha=0.6) )

plt.title('Elbow Method')

plt.xlabel('number of clusters')

plt.ylabel('wcss')

plt.show()



# It is clear from the elbow curve, 5 clusters will solve the problem
# scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
# Training the KMeans model with n_clusters=5

kmeans_model = KMeans(n_clusters=5, init='k-means++', random_state=42)

y_kmeans = kmeans_model.fit_predict(X)
y_kmeans
data_segments = pd.concat([mall_data, pd.DataFrame(y_kmeans, columns= ['Segment'])], axis= 1)

data_segments.head()


# customers on segment bases

plt.style.use('seaborn')

data_segments['Segment'].value_counts().plot(kind = 'bar')

plt.show()
# segmentwise customer data based on gender

sns.relplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)',kind = 'scatter' ,data = data_segments, hue = 'Gender', col = 'Segment')

plt.show()


# Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 30, c = 'brown', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 30, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 30, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 30, c = 'orange', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 30, c = 'red', label = 'Cluster 5')

plt.scatter(x=kmeans_model.cluster_centers_[:, 0], y=kmeans_model.cluster_centers_[:, 1], s=100, c='black', marker='+', label='Cluster Centers')

plt.legend()

plt.title('Clusters of customers')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.show()