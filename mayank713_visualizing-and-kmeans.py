# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')
data.head()
data.info()
# checking whether there are any null values

data.isnull().sum()
# plot to check spending score for men vs women

sns.barplot(data['Genre'], data['Spending Score (1-100)'])
data.info()
# we have one categorical column

# let's convert it to one hot enencoding 

data = pd.get_dummies(data)
# let's have a look at our new data

data.head()
# we don't need both the columns we can know the gender with only one columns

data = data.drop(['Genre_Female'], axis=1)
data.head()
data.drop(['CustomerID'], axis=1, inplace=True)
sns.pairplot(data, hue='Genre_Male')
X = data.iloc[:, 1:].values

X[:10,:]
from sklearn.cluster import KMeans
# elbow method to get the optimum number of clusters

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters=i, max_iter=300)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
plt.plot(wcss)

plt.xlabel('number of clusters')

plt.ylabel('Mean distance of clusters')
# looks like 5 is the optimum number of clusters

kmeans = KMeans(n_clusters=5)

y_pred = kmeans.fit_predict(X)
y_pred
# let's visualize the result

plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c='red')

plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue')

plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], c='green')

plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], c='cyan')

plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], c='magenta')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('Spending Scores vs Annual Income')

plt.show()
# let's visualize the clusters along with the gender

sns.scatterplot(X[y_pred == 0, 0], X[y_pred == 0, 1], hue=X[y_pred == 0, 2], legend=False)

sns.scatterplot(X[y_pred == 1, 0], X[y_pred == 1, 1], hue=X[y_pred == 1, 2],legend=False)

sns.scatterplot(X[y_pred == 2, 0], X[y_pred == 2, 1], hue=X[y_pred == 2, 2],legend=False)

sns.scatterplot(X[y_pred == 3, 0], X[y_pred == 3, 1], hue=X[y_pred == 3, 2],legend=False)

sns.scatterplot(X[y_pred == 4, 0], X[y_pred == 4, 1], hue=X[y_pred == 4, 2], legend=False)

sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300,legend=False)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('Spending Scores vs Annual Income for men and women')

plt.legend(['Male', 'Female'])

plt.show()