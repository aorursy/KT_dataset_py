import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans
os.chdir("/kaggle/input")

users = pd.read_csv('data_student.csv', delimiter=',')

users.head()
users.shape
users.dtypes
users.isna().sum()
users.duplicated().sum()
users.describe()
plt.figure(1 , figsize = (15 , 4))

pal1 = ["#FA5858", "#58D3F7", "#704041", "#f5c3c4"]

sns.countplot(y = ' UNS' , data = users, palette=pal1)

plt.show()
sns.set(style="ticks")

pal = ["#FA5858", "#58D3F7", "#adf2f1", "#704041", "#197a64"]



sns.pairplot(users, hue=" UNS", palette=pal)

plt.title(" UNS")
X = users[['STG' , 'PEG']].iloc[: , :].values

inertia = []

for n in range(1 , 10):

    models = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=100, 

                        tol=0.0001,  random_state= 100  , algorithm='elkan') )

    models.fit(X)

    inertia.append(models.inertia_)
plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 10) , inertia , 'o')

plt.plot(np.arange(1 , 10) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
models = (KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

models.fit(X)

labels = models.labels_

centroids = models.cluster_centers_
print(models.cluster_centers_)

print(models.inertia_)

print(models.n_iter_)
fig = plt.figure(figsize=(12,8))



plt.scatter(X[:,0], X[:,1], c=models.labels_, cmap="Set1_r", s=25)

plt.scatter(models.cluster_centers_[:,0] ,models.cluster_centers_[:,1], color='blue', marker="*", s=250)

plt.title("Kmeans Clustering \n Finding Unknown Groups in the Population", fontsize=16)

plt.show()
models2 = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

models2.fit(X)

labels2 = models2.labels_

centroids2 = models2.cluster_centers_

print(models2.cluster_centers_)
print(models2.inertia_)

print(models2.n_iter_)
fig = plt.figure(figsize=(12,8))



plt.scatter(X[:,0], X[:,1], c=models2.labels_, cmap="Set1_r", s=25)

plt.scatter(models2.cluster_centers_[:,0] ,models2.cluster_centers_[:,1], color='blue', marker="*", s=250)

plt.title("Kmeans Clustering \n Finding Unknown Groups in the Population", fontsize=16)

plt.show()
X1 = users[['STG' , 'LPR', 'PEG']].iloc[: , :].values

inertia = []

for n in range(1 , 10):

    models3 = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=100, 

                        tol=0.0001,  random_state= 100  , algorithm='elkan') )

    models3.fit(X)

    inertia.append(models3.inertia_)
plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 10) , inertia , 'o')

plt.plot(np.arange(1 , 10) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
models3 = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

models3.fit(X)

labels3 = models3.labels_

centroids3 = models3.cluster_centers_
# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff
users['labels3'] =  labels3

trace3 = go.Scatter3d(

    x= users['STG'],

    y= users['LPR'],

    z= users['PEG'],

    mode='markers',

     marker=dict(

        color = users['labels3'], 

        size= 15,

        line=dict(

            color= users['labels3'],

            width= 12

        ),

        opacity=0.8

     )

)

data = [trace3]

layout = go.Layout(

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'STG'),

            yaxis = dict(title  = 'LPR'),

            zaxis = dict(title  = 'PEG')

        )

)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
import scipy.cluster.hierarchy as shc

X = users[['STG', 'SCG', 'STR', 'LPR', 'PEG']].iloc[: , :].values

plt.figure(figsize=(10, 7))

plt.title("User Knowledge Dendograms")

plt.xlabel('Users')

dend = shc.dendrogram(shc.linkage(X, method='complete'))