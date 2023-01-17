# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from collections import Counter

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
data.head()
data.info()
from sklearn.cluster import KMeans



wcss = []



for k in range(1,50):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(scaled_df)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1,50),wcss)

plt.show()
data2=data.copy()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_df = scaler.fit_transform(data)
kmeans2 = KMeans(n_clusters=10)



clusters = kmeans2.fit_predict(scaled_df)



data2["label"]= clusters



plt.show()
data2
data3=data.copy()

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kmeans3 = KMeans(n_clusters = 10)

pipe = make_pipeline(scalar,kmeans3)

pipe.fit(data3)

labels = pipe.predict(data3)

data3["label"]= labels
data3
plt.scatter(data2.chlorides[data2.label==0],data2.pH[data2.label==0],color="red")

plt.scatter(data2.chlorides[data2.label==1],data2.pH[data2.label==1],color="yellow")

plt.scatter(data2.chlorides[data2.label==2],data2.pH[data2.label==2],color="green")

plt.scatter(data2.chlorides[data2.label==3],data2.pH[data2.label==3],color="blue")

plt.scatter(data2.chlorides[data2.label==4],data2.pH[data2.label==4],color="pink")

plt.scatter(data2.chlorides[data2.label==5],data2.pH[data2.label==5],color="orange")

plt.scatter(data2.chlorides[data2.label==6],data2.pH[data2.label==6],color="purple")

plt.scatter(data2.chlorides[data2.label==7],data2.pH[data2.label==7])

plt.scatter(data2.chlorides[data2.label==8],data2.pH[data2.label==8])

plt.scatter(data2.chlorides[data2.label==9],data2.pH[data2.label==9])

plt.scatter(data2.chlorides[data2.label==10],data2.pH[data2.label==10])







plt.scatter(data3.chlorides[data3.label==0],data3.pH[data3.label==0],color="red")

plt.scatter(data3.chlorides[data3.label==1],data3.pH[data3.label==1],color="yellow")

plt.scatter(data3.chlorides[data3.label==2],data3.pH[data3.label==2],color="green")

plt.scatter(data3.chlorides[data3.label==3],data3.pH[data3.label==3],color="blue")

plt.scatter(data3.chlorides[data3.label==4],data3.pH[data3.label==4],color="pink")

plt.scatter(data3.chlorides[data3.label==5],data3.pH[data3.label==5],color="orange")

plt.scatter(data3.chlorides[data3.label==6],data3.pH[data3.label==6],color="purple")

plt.scatter(data3.chlorides[data3.label==7],data3.pH[data3.label==7])

plt.scatter(data3.chlorides[data3.label==8],data3.pH[data3.label==8])

plt.scatter(data3.chlorides[data3.label==9],data3.pH[data3.label==9])

plt.scatter(data3.chlorides[data3.label==10],data3.pH[data3.label==10])
data3[data3["label"]==0].count()
data2[data2["label"]==0].count()
print("data 2 = ",data2[data2["label"]==0].index)

print("data 3 = ",data3[data3["label"]==0].index)

    
print("data 2 = ",data2[data2["label"]==1].index)

print("data 3 = ",data3[data3["label"]==6].index)
from scipy.cluster.hierarchy import linkage,dendrogram



merg = linkage(data.iloc[200:220,:],method = 'single')

dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)

plt.show()
data4=data.copy()

from sklearn.cluster import AgglomerativeClustering



hiyertical_cluster = AgglomerativeClustering(n_clusters =7,affinity="euclidean",linkage="ward")



cluster = hiyertical_cluster.fit_predict(data)



data4["label"]=cluster

data4
plt.scatter(data3.chlorides[data4.label==0],data3.pH[data4.label==0],color="red")

plt.scatter(data3.chlorides[data4.label==1],data3.pH[data4.label==1],color="yellow")

plt.scatter(data3.chlorides[data4.label==2],data3.pH[data4.label==2],color="green")

plt.scatter(data3.chlorides[data4.label==3],data3.pH[data4.label==3],color="blue")

plt.scatter(data3.chlorides[data4.label==4],data3.pH[data4.label==4],color="pink")

plt.scatter(data3.chlorides[data4.label==5],data3.pH[data4.label==5],color="orange")

plt.scatter(data3.chlorides[data4.label==6],data3.pH[data4.label==6],color="purple")

plt.scatter(data3.chlorides[data4.label==7],data3.pH[data4.label==7])