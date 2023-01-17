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
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')

df.sample(5)
df.describe()
df.info()
df.isna().sum()
sns.pairplot(df[df.columns.drop('CustomerID')],hue='Genre')

plt.show()
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(15,10))

print(df['Genre'].value_counts())

sns.countplot(x=df['Genre'],ax = ax[0,0],)

sns.distplot(df['Age'],bins=50,ax=ax[0,1])

sns.distplot(df['Annual Income (k$)'],bins=50,ax=ax[1,0])

sns.distplot(df['Spending Score (1-100)'],bins=50,ax=ax[1,1])

plt.show()
plt.figure(figsize=(15,5))

sns.boxplot('Annual Income (k$)','Spending Score (1-100)',data=df)

plt.show()
sns.jointplot(df['Annual Income (k$)'],df['Spending Score (1-100)'],kind = 'hex')

plt.show()
plt.figure(figsize=(15,5))

sns.boxplot('Age','Spending Score (1-100)',data=df)

plt.show()
X = df[['Annual Income (k$)','Spending Score (1-100)']].values
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS (Within Cluster Sum of Squares)')

plt.show()
# Fitting K-Means to the dataset.

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)

pred = kmeans.fit_predict(X)
# Visualising the clusters

plt.figure(figsize=(15,10))

plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[pred == 4, 0], X[pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()