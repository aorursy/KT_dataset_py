# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
print(df.shape)
print(df.info())
print(df.describe())
fig, ax = plt.subplots(figsize = (8, 6))



df['Age'].plot.hist(color = 'seagreen', rwidth = 0.8, edgecolor = 'black', linewidth = 1)

plt.title('Age Distribution of Customers')

plt.xlabel('Age')

plt.ylabel('No. of customers')

plt.show()
fig, ax = plt.subplots(figsize = (16, 5))



sns.countplot(x = 'Age', data = df, palette = 'bone')

plt.title('Age Distribution of Customers')

plt.xlabel('Age')

plt.ylabel('No. of customers')

plt.show()
fig, ax = plt.subplots(figsize = (8, 6))



df['Annual Income (k$)'].plot.hist(color = 'seagreen', rwidth = 0.8, edgecolor = 'black', linewidth = 1)

plt.title('Salary Distribution of Customers')

plt.xlabel('Annual Salary')

plt.ylabel('No. of customers')

plt.show()
fig, ax = plt.subplots(figsize = (16, 5))



sns.countplot(x = 'Annual Income (k$)', data = df, palette = 'bone')

plt.title('Salary Distribution of Customers')

plt.xlabel('Salary')

plt.ylabel('No. of customers')

plt.show()
fig, ax = plt.subplots(figsize = (8, 6))



df['Spending Score (1-100)'].plot.hist(color = 'seagreen', rwidth = 0.8, edgecolor = 'black', linewidth = 1)

plt.title('Spending Score Distribution of Customers')

plt.xlabel('Spending Score')

plt.ylabel('No. of customers')

plt.show()
fig, ax = plt.subplots(figsize = (16, 5))



sns.countplot(x = 'Spending Score (1-100)', data = df, palette = 'bone')

plt.title('Spending Score Distribution of Customers')

plt.xlabel('Spending Score')

plt.ylabel('No. of customers')

plt.show()
from sklearn.cluster import KMeans
df.head()
X = df.iloc[:, [3,4]].values
inertia = []



for i in range(1,15):

    model = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    model.fit(X)

    inertia.append(model.inertia_)
fig, ax = plt.subplots(figsize = (8, 6))

plt.plot(range(1,15), inertia)

plt.title('Inertia')

plt.xlabel('No. of Clusters')

plt.ylabel('Inertia')

plt.show()
model = KMeans(n_clusters= 5, init='k-means++', random_state=0)

Y= model.fit_predict(X)
fig, ax = plt.subplots(figsize = (12, 8))



plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s = 70, c = 'red', label = 'Moderate Income - Moderate Expenses')

plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s = 70, c = 'blue', label = 'Less Income - High Expenses')

plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s = 70, c = 'green', label = 'High Income - High Expenses')

plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s = 70, c = 'grey', label = 'Less Income - Less Expenses')

plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s = 70, c = 'brown', label = 'High Income - Less Expenses')

plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 250, c = 'black', label = 'Cluster Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
X1 = df.iloc[:, [2, 4]].values
inertia = []



for i in range(1, 15):

    model = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300)

    model.fit(X1)

    inertia.append(model.inertia_)
fig, ax = plt.subplots(figsize = (8, 6))

plt.plot(range(1,15), inertia)

plt.title('Inertia')

plt.xlabel('No. of Clusters')

plt.ylabel('Inertia')

plt.show()
model = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300)

Y1 = model.fit_predict(X1)
fig, ax = plt.subplots(figsize = (12, 8))



plt.scatter(X1[Y1 == 0, 0], X1[Y1 == 0, 1], s = 70, c = 'red', label = 'Mid Age - Moderate Expenses')

plt.scatter(X1[Y1 == 1, 0], X1[Y1 == 1, 1], s = 70, c = 'blue', label = 'Young Age - Less Expenses')

plt.scatter(X1[Y1 == 2, 0], X1[Y1 == 2, 1], s = 70, c = 'green', label = 'Old Age - Less Expenses')

plt.scatter(X1[Y1 == 3, 0], X1[Y1 == 3, 1], s = 70, c = 'grey', label = 'Mid Age - Less Expenses')

plt.scatter(X1[Y1 == 4, 0], X1[Y1 == 4, 1], s = 70, c = 'brown', label = 'Old Age - High Expenses')

plt.scatter(X1[Y1 == 5, 0], X1[Y1 == 5, 1], s = 70, c = 'yellow', label = 'Old Age - Moderate Expenses')



plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 250, c = 'black', label = 'Cluster Centroids')

plt.title('Clusters of customers')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
df.head(10)
from scipy.cluster.vq import whiten
X2 = df.iloc[:, [2, 4]].values
scaled = whiten(X2)
np.std(scaled)
inertia = []



for i in range(1, 15):

    model = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300)

    model.fit(X2)

    inertia.append(model.inertia_)
fig, ax = plt.subplots(figsize = (8, 6))

plt.plot(range(1,15), inertia)

plt.title('Inertia')

plt.xlabel('No. of Clusters')

plt.ylabel('Inertia')

plt.show()
model = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300)

Y2 = model.fit_predict(X2)
fig, ax = plt.subplots(figsize = (12, 8))



plt.scatter(X2[Y2 == 0, 0], X2[Y2 == 0, 1], s = 70, c = 'red', label = 'Old Age - Moderate Expenses')

plt.scatter(X2[Y2 == 1, 0], X2[Y2 == 1, 1], s = 70, c = 'blue', label = 'Mid Age - Moderate Expenses')

plt.scatter(X2[Y2 == 2, 0], X2[Y2 == 2, 1], s = 70, c = 'green', label = 'Less Age - Less Expenses')

plt.scatter(X2[Y2 == 3, 0], X2[Y2 == 3, 1], s = 70, c = 'grey', label = 'Young Age - Less Expenses')

plt.scatter(X2[Y2 == 4, 0], X2[Y2 == 4, 1], s = 70, c = 'brown', label = 'Old Age - High Expenses')

plt.scatter(X2[Y2 == 5, 0], X2[Y2 == 5, 1], s = 70, c = 'yellow', label = 'Less Age - Moderate Expenses')



plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 250, c = 'black', label = 'Cluster Centroids')

plt.title('Clusters of customers')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()