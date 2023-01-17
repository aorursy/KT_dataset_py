import pandas as pd

import numpy as np

from pandas import plotting

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff

import os
df = pd.read_csv("../input/Mall_Customers.csv")

dat = ff.create_table(df)

py.iplot(dat)
desc = ff.create_table(df.describe())
py.iplot(desc)
df.isnull()
plt.rcParams['figure.figsize'] = (15,10)

plotting.andrews_curves(df.drop("CustomerID",axis=1),"Gender")

plt.title("Gender Curve")

plt.show()
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (18, 8)

plt.subplot(1,2,1)

sns.set(style = 'whitegrid')

sns.distplot(df['Annual Income (k$)'],color = 'green')

plt.title('Annual Income')

plt.xlabel('Range of Annual Income')

plt.ylabel('Count')

plt.show()



plt.subplot(1,2,2)

sns.set(style = 'whitegrid')

sns.distplot(df['Age'],color = 'red')

plt.title('Age')

plt.xlabel('Range of Age')

plt.ylabel('Count')

plt.show()
labels = ['Female','Male']

size = df['Gender'].value_counts()

colors = ['lightgreen','orange']

explode = [0,0.1]



plt.rcParams['figure.figsize'] = (9,9)

plt.pie(size,colors = colors,explode = explode,labels = labels,shadow = True,autopct = "%.2f%%")

plt.title('Gender')

plt.axis('off')

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (15,8)

sns.countplot(df['Age'],palette = 'rainbow')

plt.title('Age')

plt.show()
plt.rcParams['figure.figsize'] = (20, 8)

sns.countplot(df['Annual Income (k$)'], palette = 'hsv')

plt.title('Distribution of Annual Income', fontsize = 20)

plt.show()
sns.pairplot(df)
plt.rcParams['figure.figsize'] =  (20,8)

sns.countplot(df['Spending Score (1-100)'],palette = 'Paired')
plt.rcParams['figure.figsize'] = (15, 8)

sns.heatmap(df.corr(), cmap = 'terrain', annot = True)

plt.show()
plt.rcParams['figure.figsize'] = (18,7)

sns.boxenplot(df['Gender'],df['Spending Score (1-100)'],palette = 'bright')

plt.title('Gender Vs Spending Score')

plt.show()
x = df['Annual Income (k$)']

y = df['Age']

z = df['Spending Score (1-100)']

sns.lineplot(x,y,color = 'Blue')

sns.lineplot(x,z,color = 'Pink')

plt.title('Annual Income vs Age and Spending Score')

plt.show()
x = df.iloc[:,[3,4]].values

print(x.shape)
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(x)

    wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)

plt.title('Elbow Method')

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'red', label = 'miserable')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'green', label = 'regular')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'blue', label = 'elite')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'yellow', label = 'spenders')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'magenta', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'orange' , label = 'centroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.grid()

plt.show()
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)



plt.rcParams['figure.figsize'] = (15, 5)

plt.plot(range(1, 11), wcss)

plt.title('K-Means Clustering(The Elbow Method)')

plt.xlabel('Age')

plt.ylabel('Count')

plt.grid()

plt.show()
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

ymeans = kmeans.fit_predict(x)

plt.rcParams['figure.figsize'] = (10, 10)

plt.title('Cluster of Ages', fontsize = 30)

plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'Regular Customers' )

plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')

plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')

plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black')

plt.style.use('fivethirtyeight')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.grid()

plt.show()
x = df[['Age', 'Spending Score (1-100)', 'Annual Income (k$)']].values

km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

km.fit(x)

labels = km.labels_

centroids = km.cluster_centers_

df['labels'] =  labels

trace1 = go.Scatter3d(

    x= df['Age'],

    y= df['Spending Score (1-100)'],

    z= df['Annual Income (k$)'],

    mode='markers',

     marker=dict(

        color = df['labels'], 

        size= 10,

        line=dict(

            color= df['labels'],

            width= 12

        ),

        opacity=0.8

     )

)

dv = [trace1]



layout = go.Layout(

    title = 'Character vs Gender vs Alive or not',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    ),

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)



fig = go.Figure(data = dv, layout = layout)

py.iplot(fig)