# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
#class1

x1 = np.random.normal(25,5,1000)

y1 = np.random.normal(25,5,1000)

#class2

x2 = np.random.normal(55,5,1000)

y2 = np.random.normal(60,5,1000)

#class3

x3 = np.random.normal(55,5,1000)

y3 = np.random.normal(15,5,1000)
x = np.concatenate((x1,x2,x3),axis=0)

y = np.concatenate((y1,y2,y3),axis=0)

dictionary = {"x":x,"y":y}
data = pd.DataFrame(dictionary)
plt.scatter(x1,y1)

plt.scatter(x2,y2)

plt.scatter(x3,y3)

plt.show()
plt.scatter(x1,y1,color="black")

plt.scatter(x2,y2,color="black")

plt.scatter(x3,y3,color="black")

plt.show()
from sklearn.cluster import KMeans

wcss = []

for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)

plt.xlabel("number of k (cluster) value")

plt.ylabel("wcss")

plt.show()
kmeans2 = KMeans(n_clusters=3)

clusters = kmeans2.fit_predict(data)

data["label"] = clusters

plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color = "red")

plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color = "black")

plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color = "blue")

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color = "yellow")

plt.show()