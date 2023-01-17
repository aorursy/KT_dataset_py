

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
x1 = np.random.normal(25,5,1000)

y1 = np.random.normal(25,5,1000)



x2 = np.random.normal(55,5,1000)

y2 = np.random.normal(60,5,1000)



x3 = np.random.normal(55,5,1000)

y3 = np.random.normal(15,5,1000)
x = np.concatenate((x1,x2,x3),axis=0)

y = np.concatenate((y1,y2,y3),axis=0)
dictionary = {"x":x,"y":y}
data=pd.DataFrame(dictionary)
plt.scatter(x1,y1)

plt.scatter(x2,y2)

plt.scatter(x3,y3)

plt.show()
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,15):

    kmeans=KMeans(n_clusters=k)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss)

plt.xlabel("Number of k(cluster)values")

plt.ylabel("wcss")

plt.show()
km=KMeans(n_clusters=3)

clusters=km.fit_predict(data)

data["Label"]=clusters
plt.scatter(data.x[data.Label == 0],data.y[data.Label == 0],color="red")

plt.scatter(data.x[data.Label == 1],data.y[data.Label == 1],color="green")

plt.scatter(data.x[data.Label == 2],data.y[data.Label == 2],color="blue")

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="yellow") #Centroid location

plt.show()