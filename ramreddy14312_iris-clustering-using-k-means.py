import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
data=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

data.head()
petal=data.drop(['sepal_length','sepal_width','species'],1)

petal
plt.scatter(petal['petal_length'],petal['petal_width'])
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3)

kmeans
kmeans.fit(petal)
pred=kmeans.predict(petal)

pred
centroids=kmeans.cluster_centers_

centroids
petal['CLUSTERING']=pred

petal
petal1=petal[petal.CLUSTERING==0]

petal2=petal[petal.CLUSTERING==1]

petal3=petal[petal.CLUSTERING==2]



plt.scatter(petal1['petal_length'],petal1['petal_width'],color='green')

plt.scatter(petal2['petal_length'],petal2['petal_width'],color='red')

plt.scatter(petal3['petal_length'],petal3['petal_width'],color='yellow')

plt.scatter(centroids[:,0],centroids[:,1],color='blue',marker='*')





plt.xlabel('petal_length')

plt.ylabel('petal_width')

distortions=[]

K=range(1,10)

for k in K:

    kmeans=KMeans(n_clusters=k)

    kmeans.fit(petal)

    distortions.append(kmeans.inertia_)

    

distortions
plt.plot(K,distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
data['species'].value_counts()
fulldata=data.drop('species',1)

fulldata.head()
KMNS=KMeans(n_clusters=3)

KMNS
KMNS.fit(fulldata)
prediction=KMNS.predict(fulldata)

prediction
data['predicted']=prediction

data
centroids=KMNS.cluster_centers_

centroids
data1=data.copy()

data1["species"]=data1["species"].map({'Iris-versicolor':0,'Iris-setosa':1,'Iris-virginica':2}).astype(int)

data1['predicted']=prediction

data1
data['predicted']=prediction

data
from sklearn.metrics import confusion_matrix

confusion_matrix(data1['species'],prediction)