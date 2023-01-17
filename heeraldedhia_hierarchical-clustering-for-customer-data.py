import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns



import plotly as py

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore')



from sklearn import preprocessing 

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering 

df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
df.isnull().sum()
df.describe()
plt.figure(1 , figsize = (15 , 6))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    sns.distplot(df[x] , bins = 15)

    plt.title('Distplot of {}'.format(x))

plt.show()
label_encoder = preprocessing.LabelEncoder() 



df['Gender'] = label_encoder.fit_transform(df['Gender'])

df.head()
plt.figure(1, figsize = (16 ,8))

sns.heatmap(df)

plt.show()
plt.figure(1, figsize = (16 ,8))

dendrogram = sch.dendrogram(sch.linkage(df, method  = "ward"))



plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='average')



y_hc = hc.fit_predict(df)

y_hc
df['cluster'] = pd.DataFrame(y_hc)
trace1 = go.Scatter3d(

    x= df['Age'],

    y= df['Spending Score (1-100)'],

    z= df['Annual Income (k$)'],

    mode='markers',

     marker=dict(

        color = df['cluster'], 

        size= 10,

        line=dict(

            color= df['cluster'],

            width= 12

        ),

        opacity=0.8

     )

)

data = [trace1]

layout = go.Layout(

    title= 'Clusters using Agglomerative Clustering',

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
X = df.iloc[:, [3,4]].values

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')

plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')

plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')

plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='purple', label ='Cluster 4')

plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='orange', label ='Cluster 5')

plt.title('Clusters of Customers (Hierarchical Clustering Model)')

plt.xlabel('Annual Income(k$)')

plt.ylabel('Spending Score(1-100)')

plt.show()
df.head()
df.to_csv("segmented_customers.csv", index = False)