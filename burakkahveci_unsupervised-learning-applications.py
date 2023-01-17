# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Mall_Customers.csv')
#data.head()
#data.info()
#Change from Male and Female to 0-1 
data.Gender = [1 if each == "Female" else 0 for each in data.Gender ]
data = data.rename(columns={'Annual Income (k$)':'annual_income','Spending Score (1-100)':'Spending_Score'})
data.describe()
trace1= go.Scatter(
    x= data.Age,
    y= data.Spending_Score,
    mode = 'markers',
    xaxis='x1',
    yaxis='y1',
    name="Spending Score"

)

data1=[trace1]

layout = dict(title='Comparison of Feature',
             xaxis= dict(title= 'Age',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Spending Score',ticklen= 5,zeroline= False))

iplot(go.Figure(data=data1, layout=layout))
trace2= go.Scatter(
    x= data.Age,
    y= data.annual_income,
    mode = 'markers',
    xaxis='x2',
    yaxis='y2',
    name="annual income"
)

data1=[trace2]

layout = dict(title='Comparison of Feature',
             xaxis= dict(title= 'Age',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Annual Income',ticklen= 5,zeroline= False))

iplot(go.Figure(data=data1, layout=layout))
trace3= go.Scatter(
    x= data.Spending_Score,
    y= data.annual_income,
    mode = 'markers',
    xaxis='x3',
    yaxis='y3',
    name="annual income"
)

data1=[trace3]

layout = dict(title='Comparison of Feature',
             xaxis= dict(title= 'Spending Score',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Annual Income',ticklen= 5,zeroline= False))

iplot(go.Figure(data=data1, layout=layout))
x1 = data[['Age' , 'Spending_Score']].iloc[: , :].values
x2 = data[['Age' , 'annual_income']].iloc[: , :].values
x3 = data[['annual_income' , 'Spending_Score']].iloc[: , :].values
#select K value

from sklearn.cluster import KMeans
WCSS = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x1)
    WCSS.append(kmeans.inertia_)
    
plt.plot(range(1,15),WCSS)
plt.xlabel("Number of K Value(Cluster)")
plt.ylabel("WCSS")
plt.grid()
plt.show()
kmean2 =KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
clusters = kmean2.fit_predict(x1) #create to model with k =3 #fit_predict= fitting and predict data
labels1 = kmean2.labels_
centroids1 = kmean2.cluster_centers_
h = 0.02
x_min, x_max = x1[:, 0].min() - 1, x1[:, 0].max() + 1
y_min, y_max = x1[:, 1].min() - 1, x1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmean2.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'Spending_Score' , data = data , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending_Score') , plt.xlabel('Age')
plt.grid()
plt.title("Spending Score & Age Clustering")
plt.show()
#select K value

from sklearn.cluster import KMeans
WCSS = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x2)
    WCSS.append(kmeans.inertia_)
    
plt.plot(range(1,15),WCSS)
plt.xlabel("Number of K Value(Cluster)")
plt.ylabel("WCSS")
plt.grid()
plt.show()
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(x2)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_
h = 0.02
x_min, x_max = x2[:, 0].min() - 1, x2[:, 0].max() + 1
y_min, y_max = x2[:, 1].min() - 1, x2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'annual_income' , data = data , c = labels3 , 
            s = 200 )
plt.scatter(x = centroids3[: , 0] , y =  centroids3[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Age') , plt.xlabel('Annual Income (k$)')
plt.grid()
plt.title("Age & Annual Income Clustering")
plt.show()
#select K value

from sklearn.cluster import KMeans
WCSS = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x3)
    WCSS.append(kmeans.inertia_)
    
plt.plot(range(1,15),WCSS)
plt.xlabel("Number of K Value(Cluster)")
plt.ylabel("WCSS")
plt.grid()
plt.show()
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(x3)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
h = 0.02
x_min, x_max = x3[:, 0].min() - 1, x3[:, 0].max() + 1
y_min, y_max = x3[:, 1].min() - 1, x3[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'annual_income' ,y = 'Spending_Score' , data = data , c = labels2 , 
            s = 200 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')
plt.grid()
plt.title("Spending Score & Age Clustering")
plt.show()
from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(x1,method="ward")
dendrogram(merg,leaf_rotation = 90)
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()
from sklearn.cluster import AgglomerativeClustering

HC = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean",linkage="ward")
cluster = HC.fit_predict(x1)
data["x1_label"] = cluster #add clusters to data
plt.scatter(data.Age[data.x1_label == 0],data.Spending_Score[data.x1_label == 0],color="orange")
plt.scatter(data.Age[data.x1_label == 1],data.Spending_Score[data.x1_label == 1],color="lime")
plt.scatter(data.Age[data.x1_label == 2],data.Spending_Score[data.x1_label == 2],color="red")
plt.grid()
plt.show()