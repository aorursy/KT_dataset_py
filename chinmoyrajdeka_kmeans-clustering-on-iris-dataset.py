import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
data=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

data.head()
sepal=data.drop(['petal_length','petal_width','species'],1)

sepal.head()
plt.scatter(sepal['sepal_length'],sepal['sepal_width'])
km=KMeans(n_clusters=3)

km
y_pred=km.fit_predict(sepal)

y_pred
sepal['cluster']=y_pred

sepal
sep1=sepal[sepal.cluster==0]

sep2=sepal[sepal.cluster==1]

sep3=sepal[sepal.cluster==2]



plt.scatter(sep1['sepal_length'],sep1['sepal_width'],color='green')

plt.scatter(sep2['sepal_length'],sep2['sepal_width'],color='red')

plt.scatter(sep3['sepal_length'],sep3['sepal_width'],color='yellow')



plt.xlabel('sepal_length')

plt.ylabel('sepal_width')
centroid=km.cluster_centers_

centroid
plt.scatter(sep1['sepal_length'],sep1['sepal_width'],color='green')

plt.scatter(sep2['sepal_length'],sep2['sepal_width'],color='red')

plt.scatter(sep3['sepal_length'],sep3['sepal_width'],color='yellow')

plt.scatter(centroid[:,0],centroid[:,1],color='blue',marker='*')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')
k_rng=range(1,10)

sse=[]

for k in k_rng:

    km=KMeans(n_clusters=k)

    km.fit(sepal)

    sse.append(km.inertia_)

    

sse
plt.xlabel('K')

plt.ylabel('Sum of squared error')

plt.plot(k_rng,sse)
data['species'].value_counts()
new_data=data.drop('species',1)

new_data.head()
k_rng=range(1,10)

sse=[]

for k in k_rng:

    km=KMeans(n_clusters=k)

    km.fit(new_data)

    sse.append(km.inertia_)

    

sse
plt.xlabel('K')

plt.ylabel('Sum of squared error')

plt.plot(k_rng,sse)
km=KMeans(n_clusters=3)

km
prediction=km.fit_predict(new_data)

prediction
data['predicted']=prediction

data
centroids=km.cluster_centers_

centroids
data1=data.copy()

data1["species"]=data1["species"].map({'Iris-versicolor':0,'Iris-setosa':1,'Iris-virginica':2}).astype(int)

data1['predicted']=prediction

data1
from sklearn.metrics import confusion_matrix

confusion_matrix(data1['species'],prediction)
data['predicted']=prediction

data.head()


data["predicted"]=data["predicted"].map({0:'Iris-versicolor',1:'Iris-setosa',2:'Iris-virginica'})

data