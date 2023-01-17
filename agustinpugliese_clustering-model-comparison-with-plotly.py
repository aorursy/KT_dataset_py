import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
import scipy.cluster.hierarchy as sch
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline
dataset = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
dataset.head()
dataset.describe().T
print(pd.isnull(dataset).sum())
dataset2 = dataset.copy()
dataset2 = dataset2.drop(['CustomerID'], axis = 1)
fig = px.scatter_matrix(dataset2,
    dimensions=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
    color="Gender", symbol="Gender",
    title="Scatter matrix",
    labels={col:col.replace('_', ' ') for col in dataset2.columns}) # remove underscore

fig.update_traces(diagonal_visible=False)
fig.show()
x = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
heat = go.Heatmap(z =dataset2.corr(),
                  x = x,
                  y=x,
                  xgap=1, ygap=1,
                  colorbar_thickness=20,
                  colorbar_ticklen=3,
                  hovertext = dataset2.corr(),
                  hoverinfo='text'
                   )

title = 'Correlation Matrix'               

layout = go.Layout(title_text=title, title_x=0.5, 
                   width=600, height=600,
                   xaxis_showgrid=False,
                   yaxis_showgrid=False,
                   yaxis_autorange='reversed')
   
fig=go.Figure(data=[heat], layout=layout)        
fig.show() 
hist_data = [dataset2['Age'], dataset2['Annual Income (k$)'], dataset2['Spending Score (1-100)']]
group_labels = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

fig = ff.create_distplot(hist_data, group_labels, bin_size=[5, 10, 8])
fig.update_layout(title_text='Age, Income and Score distribution')
fig.show()
fig = px.scatter(dataset2, x="Annual Income (k$)", y = "Spending Score (1-100)",size='Age', color="Gender")
fig.show()
Genre = pd.DataFrame(dataset2['Gender'].value_counts()).reset_index()
Genre.columns = ['Gender','Total']
fig = px.pie(Genre, values = 'Total', names = 'Gender', title='Gender', hole=.4, color = 'Gender',width=800, height=400)
fig.show()
fig = px.bar(Genre, x = 'Gender', y='Total', color='Gender',width=600, height=500)
fig.show()
Male = dataset2[dataset2["Gender"] == 'Male'][['Gender','Age']]
temp = pd.DataFrame(Male['Age'].value_counts().reset_index())
temp.columns = ['Age','Total']

Female = dataset2[dataset2["Gender"] == 'Female'][['Gender','Age']]
temp2 = pd.DataFrame(Female['Age'].value_counts().reset_index())
temp2.columns = ['Age','Total']
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(
    x = temp['Age'],
    y = temp['Total'],
    name='Male',
    marker_color='rgba(94, 144, 175, 0.8)'
))
fig.add_trace(go.Bar(
    x = temp2['Age'],
    y = temp2['Total'],
    name='Female',
    marker_color='rgba(249, 70, 10, 0.9)'
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(title = 'Age per genre', barmode = 'group', xaxis_tickangle=-45)
fig.show()

X = dataset2.iloc[:,2:4].values
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500, n_init = 10, random_state = 123)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
fig = go.Figure(data = go.Scatter(x = [1,2,3,4,5,6,7,8,9,10], y = wcss))


fig.update_layout(title='WCSS vs. Cluster number',
                   xaxis_title='Clusters',
                   yaxis_title='WCSS')
fig.show()
kmeans = KMeans(n_clusters = 5, init="k-means++", max_iter = 500, n_init = 10, random_state = 123)
identified_clusters = kmeans.fit_predict(X)


data_with_clusters = dataset2.copy()
data_with_clusters['Cluster'] = identified_clusters
fig = px.scatter_3d(data_with_clusters, x = 'Age', y='Annual Income (k$)', z='Spending Score (1-100)',
              color='Cluster', opacity = 0.8, size='Age', size_max=30)
fig.show()
fig = ff.create_dendrogram(X,
                           linkagefun = lambda x: sch.linkage(x, "ward"),)

# Ward minimizes the variance of the points inside a cluster.

fig.update_layout(title = 'Hierarchical Clustering', xaxis_title='Customers',
                   yaxis_title='Euclidean Distance', width=700, height=700)

fig.show()
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
identified_clusters = hc.fit_predict(X)

data_with_clusters = dataset2.copy()
data_with_clusters['Cluster'] = identified_clusters

fig = px.scatter_3d(data_with_clusters, x = 'Age', y='Annual Income (k$)', z='Spending Score (1-100)',
              color='Cluster', opacity = 0.8, size='Age', size_max=30)
fig.show()
ap = AffinityPropagation(random_state = 0)
identified_clusters = ap.fit_predict(X)

data_with_clusters = dataset2.copy()
data_with_clusters['Cluster'] = identified_clusters

fig = px.scatter_3d(data_with_clusters, x = 'Age', y='Annual Income (k$)', z='Spending Score (1-100)',
              color='Cluster', opacity = 0.8, size='Age', size_max=30)
fig.show()
DBS = DBSCAN(eps = 9, min_samples = 5)

identified_clusters = DBS.fit_predict(X)

data_with_clusters = dataset2.copy()
data_with_clusters['Cluster'] = identified_clusters

fig = px.scatter_3d(data_with_clusters, x = 'Age', y='Annual Income (k$)', z='Spending Score (1-100)',
              color='Cluster', opacity = 0.8, size='Age', size_max=30)
fig.show()