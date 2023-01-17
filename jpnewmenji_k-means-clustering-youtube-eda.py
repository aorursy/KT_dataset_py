# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.graph_objs import *
import seaborn as sns
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/USvideos.csv')
df.head()
df.dtypes
plt.figure(figsize=(12,8))
sns.countplot(x="category_id", data=df)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Video Category', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Youtube  Category", fontsize=15)
plt.show()
from ggplot import *
ggplot(aes(x='likes', y='comment_count'), data=df) + \
    geom_point(color='steelblue', size=1) + \
    stat_smooth()
from ggplot import *
ggplot(aes(x='likes', y='dislikes'), data=df) + \
    geom_point(color='steelblue', size=1) + \
    stat_smooth()
cluster = df[['likes','dislikes', 'views', 'comment_count']]
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(cluster)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
df2 = cluster.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(df2)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0) 
y_kmeans = kmeans.fit_predict(df2)
df2['cluster']=y_kmeans

trace1 = go.Scatter3d(
    x = df2['likes'].values,
    y = df2['comment_count'].values,
    z = df2['views'].values,
    mode='markers',
    marker=dict(
        size=12,
        color=df2['cluster'].values,# set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
    
)

data = [trace1]
layout = go.Layout(
    scene=Scene(
        xaxis=XAxis(title='Likes'),
        yaxis=YAxis(title='Comment'),
        zaxis=ZAxis(title='Views')
        ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')

