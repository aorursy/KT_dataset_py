# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
x1=np.random.normal(25,5,100)
y1=np.random.normal(25,5,100)

x2=np.random.normal(55,5,100)
y2=np.random.normal(60,5,100)

x3=np.random.normal(55,5,100)
y3=np.random.normal(15,5,100)

x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((y1,y2,y3),axis=0)

dictionary={"x":x,"y":y}
data=pd.DataFrame(dictionary)
data.describe()
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()
from scipy.cluster.hierarchy import linkage,dendrogram
merg=linkage(data,method="ward")
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()
#3 clusters are appropriate
from sklearn.cluster import AgglomerativeClustering
#Creating cluster
hierarchical_cluster=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
cluster=hierarchical_cluster.fit_predict(data)

data["label"]=cluster
plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red")
plt.scatter(data.x[data.label==1],data.y[data.label==1],color="green")
plt.scatter(data.x[data.label==2],data.y[data.label==2],color="blue")
plt.show()