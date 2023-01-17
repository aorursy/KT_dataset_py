import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head()
df.shape
df.describe()
df.info()
df.dtypes
df.isnull().sum()
sns.countplot(x='Gender', data = df)
sns.distplot(df['Age'])
sns.distplot(df['Annual Income (k$)'])
sns.distplot(df['Spending Score (1-100)'])
sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender')

plt.show()
sns.violinplot(x=df['Gender'], y=df['Annual Income (k$)'])
sns.swarmplot(x=df['Gender'], y=df['Spending Score (1-100)'])
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(df.iloc[:,1].values)

df['Gender'] = integer_encoded
df.head()
sns.heatmap(df.iloc[:, 1:5].corr(), annot=True, cmap='coolwarm')
X = df.iloc[:, [3, 4]].values
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of cluster')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)

y_kmeans
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'blue', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'pink', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 20, c = 'orange', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'red', label = 'Centroids')

plt.title('Customer segments')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.show()