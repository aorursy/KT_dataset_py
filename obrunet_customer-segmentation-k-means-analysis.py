import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
df = pd.read_csv(r'../input/Mall_Customers.csv')

df.head()
df.shape
df.info()
df.duplicated().sum()
df.describe()
plt.figure(1, figsize=(16,10))

sns.pairplot(data=df, hue='Gender')

plt.show()
plt.figure(1, figsize=(4,4))

sns.countplot(x='Gender', data=df)

plt.show()
plt.figure(1, figsize=(16,4))

n = 0 

for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1, 3, n)

    plt.subplots_adjust(hspace=0.5 , wspace=0.5)

    sns.distplot(df[x] , bins=10)

    plt.title('Distplot of {}'.format(x))

plt.show()
X = df.iloc[:, -2:]
km_inertias, km_scores = [], []



for k in range(2, 11):

    km = KMeans(n_clusters=k).fit(X)

    km_inertias.append(km.inertia_)

    km_scores.append(silhouette_score(X, km.labels_))

    

sns.lineplot(range(2, 11), km_inertias)

plt.title('elbow graph / inertia depending on k')

plt.show()
sns.lineplot(range(2, 11), km_scores)

plt.title('scores depending on k')

plt.show()
X = df.iloc[:, -3:]
km_inertias, km_scores = [], []



for k in range(2, 11):

    km = KMeans(n_clusters=k).fit(X)

    km_inertias.append(km.inertia_)

    km_scores.append(silhouette_score(X, km.labels_))

    

sns.lineplot(range(2, 11), km_inertias)

plt.title('elbow graph / inertia depending on k')

plt.show()
sns.lineplot(range(2, 11), km_scores)

plt.title('scores depending on k')

plt.show()
km = KMeans(n_clusters=5).fit(X)
# K-Means visualization on pair of 2 features

plt.figure(figsize=(10, 6))

sns.scatterplot(X.iloc[:, 1], X.iloc[:, 2], hue=km.labels_)

plt.show()
# K-Means visualization on another pair of 2 features

plt.figure(figsize=(10, 6))

sns.scatterplot(X.iloc[:, 0], X.iloc[:, 1], hue=km.labels_)

plt.show()
# K-Means visualization on the last pair of 2 features

plt.figure(figsize=(10, 6))

sns.scatterplot(X.iloc[:, 0], X.iloc[:, 2], hue=km.labels_)

plt.show()
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(8,8))

ax = Axes3D(fig)



xs = X.iloc[:, 0]

ys = X.iloc[:, 1]

zs = X.iloc[:, 2]

ax.scatter(xs, ys, zs, s=50, alpha=0.6, c=km.labels_)



ax.set_xlabel('Age')

ax.set_ylabel('Annual income')

ax.set_zlabel('Spending score')



plt.show()
# Profiles of customers

X['label'] = km.labels_
X.label.value_counts()
for k in range(5):

    print(f'cluster nb : {k}')

    print(X[X.label == k].describe().iloc[[0, 1, 3, 7], :-1])

    print('\n\n')
X[X.label == 1].describe().iloc[[0, 1, 3, 7], :-1]