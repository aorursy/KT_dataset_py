import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans

import warnings

import os

warnings.filterwarnings("ignore")

#py.offline.init_notebook_mode(connected = True)
df = pd.read_csv("../input/Mall_Customers.csv")

df.head()
df.shape
df.info()
df.describe()
df.dtypes
df.isnull().sum()
plt.style.use("fivethirtyeight")
plt.figure(1, figsize = (15, 6))

n = 0

for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1, 3, n)

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    sns.distplot(df[x], bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
plt.figure(1 , figsize = (15 , 5))

sns.countplot(y = 'Gender', data =df)

plt.show()
#Ploting the Relation between Age , Annual Income and Spending Score¶

plt.figure(1, figsize = (15, 7))

n = 0

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

        n += 1

        plt.subplot(3, 3, n)

        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

        sns.regplot(x = x, y = y, data = df)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
plt.figure(1, figsize = (15, 6))

for gender in ['Male', 'Female']:

    plt.scatter(x = 'Age', y = 'Annual Income (k$)', data = df[df['Gender'] == gender],s = 200, alpha=0.5, label = gender )

plt.xlabel('Age'), plt.ylabel('Annual Income (k$)')

plt.title('Age VS Annual Income w.r.t Gender')

plt.legend()

plt.show()
plt.figure(1, figsize = (15, 6))

for gender in ['Male', 'Female']:

    plt.scatter(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', data = df[df['Gender'] == gender], s = 200, alpha=0.5 , label = gender)

plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)')

plt.title('Annual Income vs Spening Score w.r.t Gender')

plt.show()
#Distribution of values in Age , Annual Income and Spending Score according to Gender

plt.figure(1, figsize = (15, 7))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1, 3, n)

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5 )

    sns.violinplot(x = x, y = 'Gender', data = df, palette ='vlag' )

    sns.swarmplot(x = x, y = 'Gender', data = df )

    plt.ylabel('Gendr')

    plt.title('Viloinplots vs Swarmplots')

plt.show()
#Clustering using K- means

#Segmentation using Age , Annual Income and Spending Score

new_data = df[['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']].iloc[:, :].values

inertia = []

for n in range(1, 11):

    algorithm = (KMeans(n_clusters = n, init='k-means++', n_init = 10, max_iter=300, tol = 0.0001, random_state = 111, algorithm = 'elkan'))

    algorithm.fit(new_data)

    inertia.append(algorithm.inertia_)
plt.figure(1, figsize = (15, 6))

plt.plot(np.arange(1, 11), inertia, 'o')

plt.plot(np.arange(1,11), inertia, '-', alpha = 0.5)

plt.xlabel('Number of cluster'), plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 6, init='k-means++', n_init = 10, max_iter=300, tol = 0.0001, random_state = 111, algorithm = 'elkan'))

algorithm.fit(new_data)

labels = algorithm.labels_

centroids = algorithm.cluster_centers_
df['labels'] = labels

trace = go.Scatter3d(x = df['Age'], y = df['Annual Income (k$)'], z = df['Spending Score (1-100)'], mode = 'markers',

                    marker = dict(color = df['labels'], size = 20, line = dict(color = df['labels'], width = 12), opacity = 0.8))

data = [trace]

layout = go.Layout(title = 'Clusters', scene = dict(xaxis = dict(title = 'Age'), yaxis = dict(title = 'Annual Income'), zaxis = dict(title = 'Spending Score')))

fig = go.Figure(data = data, layout = layout)

fig