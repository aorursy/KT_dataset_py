import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.offline as py
import plotly.figure_factory as ff
%matplotlib inline
df=pd.read_csv('../input/Mall_Customers.csv')
df.head()
df.info()
df.describe()
print("Mean of Annual Income (k$) of Female:",df['Annual Income (k$)'].loc[df['Gender'] == 'Female'].mean())
print("Mean of Annual Income (k$) of Male:",df['Annual Income (k$)'].loc[df['Gender'] == 'Male'].mean())
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,cmap='hsv',fmt='.2f',linewidths=2)
plt.show()
df.groupby('Gender').mean()
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
col = "Gender"
grouped = df[col].value_counts().reset_index()
grouped = grouped.rename(columns = {col : "count", "index" : col})

## plot
colors = ['gold', 'mediumturquoise']
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0],marker=dict(colors=colors, line=dict(color='#000000', width=2)))
layout = {'title': 'Gender(Male, Female)'}
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)
%matplotlib inline
plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
sns.distplot(df['Age'])
plt.title('Distplot of Age')
plt.subplot(1,3,2)
sns.distplot(df['Spending Score (1-100)'],hist=False)
plt.title('Distplot of Spending Score (1-100)')
plt.subplot(1,3,3)
sns.distplot(df['Annual Income (k$)'])
plt.title('Annual Income (k$)')
plt.show()
x=df
col='Age'
v1=x[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
trace1 = go.Bar(x=v1[col], y=v1["count"], name="0", marker=dict(color="rgb(63, 72, 204)"))
y = [trace1]
layout={'title':"Age count ",'xaxis':{'title':"Age"},'yaxis':{'title':"Count"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='presentation'
iplot(fig)
d1=x[x['Gender']=='Male']
d2=x[x['Gender']=='Female']
col='Age'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Male", marker=dict(color="#ff7f0e"))
trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Female", marker=dict(color='#a678de'))
y = [trace1, trace2]
layout={'title':"Age count [[ Male vs Female ]] ",'xaxis':{'title':"Age"},'yaxis':{'title':"Count"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='presentation'
iplot(fig)
col='Spending Score (1-100)'
v2=x[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v2[col], y=v2["count"], name="Emb",  marker=dict(color="#e377c2"))
layout={'title':"Spending Score (1-100)",'xaxis':{'title':"spending score"},'yaxis':{'title':"Count"}}
fig = go.Figure(data=[trace1], layout=layout)
fig.layout.template='presentation'
iplot(fig)
col='Spending Score (1-100)'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Male", marker=dict(color="#d62728"),text= df['Age'])
trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Female", marker=dict(color='rgb(63, 72, 204)'),text= df['Age'])
y = [trace1, trace2]
layout={'title':"Spending score [[ Male vs Female ]] with their Age",'xaxis':{'title':"spending acore"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='plotly_dark'
iplot(fig)
col='Annual Income (k$)'
v2=x[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v2[col], y=v2["count"], name="Emb",  marker=dict(color="#6ad49b"))
layout={'title':"Annual Income in k$",'xaxis':{'title':"Anual income"},'yaxis':{'title':"Count"}}
fig = go.Figure(data=[trace1], layout=layout)
fig.layout.template='presentation'
iplot(fig)
d1=x[x['Gender']=='Male']
d2=x[x['Gender']=='Female']
col='Spending Score (1-100)'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Male", marker=dict(color="#17becf"),text= df['Annual Income (k$)'])
trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Female", marker=dict(color='#bcbd22'),text= df['Annual Income (k$)'])
y = [trace1, trace2]
layout={'title':"Spending score [[ Male vs Female ]] with their Anual income in k$",'xaxis':{'title':"spending acore"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='plotly_dark'
iplot(fig)
col='Annual Income (k$)'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Male", marker=dict(color="#17becf"),text= df['Spending Score (1-100)'])
trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Female", marker=dict(color='#d62728'),text= df['Spending Score (1-100)'])
y = [trace1, trace2]
layout={'title':"Anual income[Male vs Female] with Spending score ",'xaxis':{'title':"Anual income"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='presentation'
iplot(fig)
col='Annual Income (k$)'
col1='Spending Score (1-100)'
v1=x[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=x[col1].value_counts().reset_index()
v2=v2.rename(columns={col1:'count1','index':col1})
v2['percent1']=v2['count1'].apply(lambda x : 100*x/sum(v2['count1']))
v2=v2.sort_values(col1)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Anual income", marker=dict(color="#6ad49b"))
trace2 = go.Scatter(x=v2[col1], y=v2["count1"], name="spending score", marker=dict(color='rgb(63, 72, 204)'))
y = [trace1, trace2]
layout={'title':"Anual income,Spending score [Male vs Female]",'xaxis':{'title':"spending score // Anual income "}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='presentation'
iplot(fig)
%matplotlib inline
import scipy.cluster.hierarchy as sch
X=df.iloc[:, [3,4]].values
plt.figure(figsize=(25,12))
dendrogram=sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendrogram plot')
plt.show()
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Curve')
plt.show() 
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)
#Visualizing all the clusters 
plt.figure(figsize=(8,5))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()