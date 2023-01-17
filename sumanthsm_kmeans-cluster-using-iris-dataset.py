# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing all required Datasets

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns #iris datsset is available in this library
data=pd.read_csv('../input/iris-datasets/Iris.csv',index_col=False)

data.head() #here is my data
data=data.set_index('Id')

data.head() #set a id colmn as index
#using 2 columns sepalwidth and sepallength

#creating scattered plot from data

plt.scatter(data[['SepalLengthCm']],data[['SepalWidthCm']])

#in the below grapgh we can say randomly that having 4 clusters
#basically in this function iam trying to find numbers of centroids or clusters 

km=KMeans()

k_range=range(1,10)

sse=[]

for k in k_range:

    km=KMeans(n_clusters=k)

    km.fit(data[['SepalLengthCm','SepalWidthCm']])

    sse.append(km.inertia_)

    

sse #arrays
#plot the k_range and sse

plt.plot(k_range,sse)
#Applying the KMean algorithm

km=KMeans(n_clusters=3)

y_pred=km.fit_predict(data[['SepalLengthCm','SepalWidthCm']])#fit and prediction on both sepal and petal

y_pred
#after creating arrays iam creating new column CLUSTER for arrays

data['cluster']=y_pred

data.head()
km.cluster_centers_
#create dataframe for three clusters

df1=data[data.cluster==0]

df2=data[data.cluster==1]

df3=data[data.cluster==2]





#plot the graph of dataframes with different colors

plt.scatter(df1['SepalLengthCm'],df1['SepalWidthCm'],c='r')

plt.scatter(df2['SepalLengthCm'],df2['SepalWidthCm'],c='g')

plt.scatter(df3['SepalLengthCm'],df3['SepalWidthCm'],c='b')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],c='black',marker='*')





plt.ylabel('sepalWidth')

plt.xlabel('sepalLength')

plt.show()
#df1.to_csv('Iris_IfCluster=0.csv')

#df2.to_csv('Iris_IfCluster=1.csv')

#df3.to_csv('Iris_IfCluster=2.csv')

#df1

#df2

#df3

#len(df1),len(df2),len(df3)