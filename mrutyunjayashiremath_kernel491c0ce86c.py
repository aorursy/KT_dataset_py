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
#Elbow Method for optimal value of k in KMeans

#Prerequisites: K-Means Clustering



#A fundamental step for any unsupervised algorithm is to determine the optimal number of clusters 

#into which the data may be clustered. The Elbow Method is one of the most popular methods to determine this optimal value of k.
from sklearn.cluster import KMeans

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

%matplotlib inline
#this is simple program KMeans easy understand 

import pandas as pd

df = pd.read_csv("../input/income.csv")
#thsi is simple program KMeans how to apply elbow method 

df.head()
plt.scatter(df['Age'],df['Income($)'])
km=KMeans(n_clusters=3)

km
y_predicted=km.fit_predict(df[['Age','Income($)']])

y_predicted
df['cluster']=y_predicted

df.head()
df1=df[df.cluster==0]

df2=df[df.cluster==1]

df3=df[df.cluster==3]



plt.scatter(df1.Age,df1['Income($)'],color="red")

plt.scatter(df2.Age,df2['Income($)'],color="blue")

plt.scatter(df3.Age,df3['Income($)'],color="green")



plt.xlabel("Age")

plt.ylabel('Income($)')

plt.legend()
scaler=MinMaxScaler()

scaler.fit(df[['Income($)']])

df['Income($)']=scaler.transform(df[['Income($)']])

df
scaler.fit(df[['Age']])

df['Age']=scaler.transform(df[['Age']])

df
km=KMeans(n_clusters=3)

y_predicted=km.fit_predict(df[['Age','Income($)']])

y_predicted

df['cluster']=y_predicted

#df.drop(['cluster'],inplace=True)

df.head()
df1=df[df.cluster==0]

df2=df[df.cluster==1]

df3=df[df.cluster==2]





plt.scatter(df1.Age,df1['Income($)'],color="red")

plt.scatter(df2.Age,df2['Income($)'],color="blue")

plt.scatter(df3.Age,df3['Income($)'],color='green')



plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker="*",label='c')

plt.legend()
kreg=range(1,10)

sse=[]

for k in kreg:

    km=KMeans(n_clusters=k)

    km.fit(df[['Age','Income($)']])

    sse.append(km.inertia_)
sse
#elbow method found 3 cluster 

plt.xlabel('K')

plt.ylabel('sum of squared error')

plt.plot(kreg,sse)