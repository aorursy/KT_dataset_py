# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import plotting
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df
df.shape
df.info()
df.describe()
df.dtypes
df.isnull().sum()
df.corr()
sns.heatmap(df.corr(),annot=True,fmt='.1f')
plt.show()
df.drop('CustomerID',axis=1,inplace=True)
df.head()
df['Gender'].value_counts()
sns.countplot(df['Gender'])
labels = ['Female', 'Male']
size = df['Gender'].value_counts()
colors = ['lightgreen', 'orange']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
plotting.andrews_curves(df, "Gender")
plt.title('Andrew Curves for Gender', fontsize = 30)
plt.show()
plt.subplot(1, 2,1)
sns.distplot(df['Annual Income (k$)'],color='green')
plt.title('Distribution of Annual Income', fontsize = 20)
plt.xlabel('Range of Annual Income')
plt.ylabel('Count')


plt.subplot(1, 2,2)
sns.distplot(df['Age'], color = 'red')
plt.title('Distribution of Age', fontsize = 20)
plt.xlabel('Range of Age')
plt.ylabel('Count')
plt.show()
plt.rcParams['figure.figsize'] = (20, 8)
sns.countplot(df['Age'])
plt.title('Distribution of Age', fontsize = 30)
plt.show()
plt.rcParams['figure.figsize'] = (25, 10)
sns.countplot(df['Annual Income (k$)'])
plt.title('Distribution of Annual Income', fontsize = 30)
plt.show()
plt.rcParams['figure.figsize'] = (25, 10)
sns.countplot(df['Spending Score (1-100)'])
plt.title('Distribution of Spending Score', fontsize = 30)
plt.show()
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df[df['Gender'] == gender] ,
                s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 
plt.title('Age vs Annual Income w.r.t Gender',fontsize=20)
plt.legend()
plt.show()
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
                data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 
plt.title('Annual Income vs Spending Score w.r.t Gender',fontsize=20)
plt.legend()
plt.show()
x = df['Annual Income (k$)']
y = df['Age']
z = df['Spending Score (1-100)']

sns.lineplot(x, y, color = 'blue')
sns.lineplot(x, z, color = 'pink')
plt.title('Annual Income vs Age and Spending Score', fontsize = 20)
plt.show()
df.corr()
x = df[['Annual Income (k$)','Spending Score (1-100)']].values
# x
x.shape
from sklearn.cluster import KMeans

SSD=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km=km.fit(x)
    SSD.append(km.inertia_)
    print(km.inertia_)
m=range(1,11)

plt.plot(m,SSD,'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('The Elbow Method', fontsize = 20)
plt.show()
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)
y_means
df['class']=y_means
df
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 70, c = 'red', label = 'Category 1')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 70, c = 'blue', label = 'Category 2')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 70, c = 'green', label = 'Category 3')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 70, c = 'magenta', label = 'Category 4')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 70, c = 'orange', label = 'Category 5')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'Centroid')


plt.style.use('fivethirtyeight')
plt.title('Segmentation using Annual Income and Spending Score', fontsize = 30)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()

from yellowbrick.cluster import KElbowVisualizer
df1 = df[['Age', 'Spending Score (1-100)']].values
algorithm = KElbowVisualizer(KMeans(init='k-means++',algorithm='elkan'), k=12, metric="distortion")
algorithm.fit(df1)
algorithm.show()
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', algorithm='elkan') )
algorithm.fit(df1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_
y_m = algorithm.fit_predict(df1)
plt.scatter(df1[y_m == 0, 0], df1[y_m == 0, 1], s = 70, c = 'red', label = 'Category 1')
plt.scatter(df1[y_m == 1, 0], df1[y_m == 1, 1], s = 70, c = 'blue', label = 'Category 2')
plt.scatter(df1[y_m == 2, 0], df1[y_m == 2, 1], s = 70, c = 'green', label = 'Category 3')
plt.scatter(df1[y_m == 3, 0], df1[y_m == 3, 1], s = 70, c = 'magenta', label = 'Category 4')

plt.scatter(centroids1[:,0], centroids1[:, 1], s = 150, c = 'black' , label = 'Centroid')

plt.rcParams["figure.figsize"] = (18,8)
plt.style.use('fivethirtyeight')
plt.xlabel('Age')
plt.ylabel('Spending_Score')
plt.title('Segmentation using Age and Spending Score')
plt.legend()
plt.grid()
plt.show()
df2 = df[['Age', 'Annual Income (k$)']].values
algorithm = KElbowVisualizer(KMeans(init='k-means++',algorithm='elkan'), k=12, metric="distortion")
algorithm.fit(df1)
algorithm.show()
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', algorithm='elkan') )
algorithm.fit(df1)
labels1 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
y_m = algorithm.fit_predict(df2)
plt.scatter(df2[y_m == 0, 0], df2[y_m == 0, 1], s = 70, c = 'red', label = 'Category 1')
plt.scatter(df2[y_m == 1, 0], df2[y_m == 1, 1], s = 70, c = 'blue', label = 'Category 2')
plt.scatter(df2[y_m == 2, 0], df2[y_m == 2, 1], s = 70, c = 'green', label = 'Category 3')
plt.scatter(df2[y_m == 3, 0], df2[y_m == 3, 1], s = 70, c = 'magenta', label = 'Category 4')

plt.scatter(centroids2[:,0], centroids2[:, 1], s = 150, c = 'black' , label = 'Centroid')

plt.rcParams["figure.figsize"] = (18,8)
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.title('Segmentation using Age and Annual Score')
plt.legend()
plt.grid()
plt.show()
df3 = df[['Age','Annual Income (k$)', 'Spending Score (1-100)']].values
algorithm = KElbowVisualizer(KMeans(init='k-means++',algorithm='elkan'), k=12, metric="distortion")
algorithm.fit(df3)
algorithm.show()
import plotly.graph_objs as go
import plotly as py
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', algorithm='elkan') )
algorithm.fit(df1)
labels1 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
y_m = algorithm.fit_predict(df3)
df['Cluster'] =  labels1
trace1 = go.Scatter3d(
    x= df['Age'],
    y= df['Spending Score (1-100)'],
    z= df['Annual Income (k$)'],
    mode='markers',
     marker=dict(
        color = df['Cluster'], 
        size= 30,
        line=dict(
            color= df['Cluster'],
            width= 18
        ),
        opacity=0.8
     )
)
data1 = [trace1]
layout = go.Layout(
title= 'Clusters in 3-D',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
        )
)
fig = go.Figure(data=data1, layout=layout)
py.offline.iplot(fig)
import scipy.cluster.hierarchy as sch
# x
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Eucledian distance")
plt.show()
from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")

y=hc.fit_predict(x)
y
plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')


plt.title('Hierarchical Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()