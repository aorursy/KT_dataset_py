# check if the plotly is installed or not

!pip install plotly
# import the libraries

import numpy as np

import pandas as pd 

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objs as go

from plotly import tools

from plotly.subplots import make_subplots

import plotly.offline as py
# load the cutomer data

df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
df.info()
# creating a two dimentional matrix 

x = df.iloc[:,[3,4]].values
# find the optimal number of clusters using elbow method



WCSS = []

for i in range(1,11):

    model = KMeans(n_clusters = i,init = 'k-means++')

    model.fit(x)

    WCSS.append(model.inertia_)

fig = plt.figure(figsize = (7,7))

plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'green')

plt.xticks(np.arange(11))

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()

# since elbow occured at 5, hence the optimal number of clusters for the mall data is 5 



model = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)

y_clusters = model.fit_predict(x)
plt.figure(figsize = (20,10))

plt.scatter(x[y_clusters == 0,0],x[y_clusters == 0,1],s = 50, c = 'green', label = "High income - Less spending")

plt.scatter(x[y_clusters == 1,0],x[y_clusters == 1,1],s = 50, c = 'blue', label = "medium income - medium spending")

plt.scatter(x[y_clusters == 2,0],x[y_clusters == 2,1],s = 50, c = 'black', label = "Hign income - high spending")

plt.scatter(x[y_clusters == 3,0],x[y_clusters == 3,1],s = 50, c = 'red', label = "Less income - high spending")

plt.scatter(x[y_clusters == 4,0],x[y_clusters == 4,1],s = 50, c = 'pink', label = "Less income and less spending")

plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1], s = 100, c = "yellow", label = "centroids")

plt.xlabel("Anual income(k$) -- >")

plt.ylabel("spending score out of 100 -- >")

plt.legend()

plt.show()
# input matrix for segmentation

x = df[['Age','Annual Income (k$)','Spending Score (1-100)']].values
# find the optimal number of clusters using elbow method  -- >This is for 3 features = [age,anual income,spending score]



WCSS = []

for i in range(1,11):

    model = KMeans(n_clusters = i,init = 'k-means++')

    model.fit(x)

    WCSS.append(model.inertia_)

fig = plt.figure(figsize = (7,7))

plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'red')

plt.xticks(np.arange(11))

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
# finding the clusters based on input matrix "x"

model = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)

y_clusters = model.fit_predict(x)
# countplot to check the number of clusters and number of customers in each cluster

sns.countplot(y_clusters)
print(x[y_clusters == 0,0][1])

print(x[y_clusters == 0,1][1])

print(x[y_clusters == 0,2][1])
# 3d scatterplot using matplotlib



fig = plt.figure(figsize = (15,15))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[y_clusters == 0,0],x[y_clusters == 0,1],x[y_clusters == 0,2], s = 40 , color = 'blue', label = "cluster 0")

ax.scatter(x[y_clusters == 1,0],x[y_clusters == 1,1],x[y_clusters == 1,2], s = 40 , color = 'orange', label = "cluster 1")

ax.scatter(x[y_clusters == 2,0],x[y_clusters == 2,1],x[y_clusters == 2,2], s = 40 , color = 'green', label = "cluster 2")

ax.scatter(x[y_clusters == 3,0],x[y_clusters == 3,1],x[y_clusters == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")

ax.scatter(x[y_clusters == 4,0],x[y_clusters == 4,1],x[y_clusters == 4,2], s = 40 , color = 'purple', label = "cluster 4")

ax.set_xlabel('Age of a customer-->')

ax.set_ylabel('Anual Income-->')

ax.set_zlabel('Spending Score-->')

ax.legend()

plt.show()

# 3d scatterplot using plotly

Scene = dict(xaxis = dict(title  = 'Age -->'),yaxis = dict(title  = 'Spending Score--->'),zaxis = dict(title  = 'Annual Income-->'))



# model.labels_ is nothing but the predicted clusters i.e y_clusters

labels = model.labels_

trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers',marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))

layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)

data = [trace]

fig = go.Figure(data = data, layout = layout)

fig.show()