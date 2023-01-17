# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path="../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv"

customer_data=pd.read_csv(file_path)
customer_data.head()
customer_data.describe()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
plt.figure(1,figsize=(12,6))

plt.subplot(1,3,1)

sns.distplot(a=customer_data["Age"],kde=False)

plt.subplot(1,3,2)

sns.distplot(a=customer_data["Annual Income (k$)"],kde=False)

plt.subplot(1,3,3)

sns.distplot(a=customer_data["Spending Score (1-100)"],kde=False)
plt.figure(1,figsize=(14,5))

plt.subplot(1,3,1)

sns.regplot(x=customer_data["Age"],y=customer_data["Annual Income (k$)"])

plt.subplot(1,3,2)

sns.regplot(x=customer_data["Age"],y=customer_data["Spending Score (1-100)"]) 

plt.subplot(1,3,3)

sns.regplot(x=customer_data["Spending Score (1-100)"],y=customer_data["Annual Income (k$)"])
X=customer_data.iloc[:,[3,4]].values


from sklearn.cluster import KMeans

cluno=[]

for i in range (1,11):

    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=100,n_init=10,random_state=0)

    kmeans.fit(X)

    cluno.append(kmeans.inertia_)

plt.plot(range(1,11),cluno)

plt.xlabel("no of clusters")

plt.ylabel("cluno ")

plt.show()
kmeans=KMeans(n_clusters=5,init="k-means++",max_iter=100,n_init=10,random_state=0)

y=kmeans.fit_predict(X)
plt.scatter(X[y==0,0],X[y==0,1],s=100,c="blue",label="normal people")

plt.scatter(X[y==1,0],X[y==1,1],s=100,c="green",label="target people")

plt.scatter(X[y==2,0],X[y==2,1],s=100,c="cyan",label="rich people")

plt.scatter(X[y==3,0],X[y==3,1],s=100,c="yellow",label="neglect people")

plt.scatter(X[y==4,0],X[y==4,1],s=100,c="grey",label="simple people")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="red",label="centroid")

plt.title("clients clusters")

plt.xlabel("annual income")

plt.ylabel("spending score")

plt.legend()

plt.show
X=customer_data.iloc[:,[2,3]].values
from sklearn.cluster import KMeans

cluno=[]

for i in range (1,11):

    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=100,n_init=10,random_state=0)

    kmeans.fit(X)

    cluno.append(kmeans.inertia_)

plt.plot(range(1,11),cluno)

plt.xlabel("no of clusters")

plt.ylabel("cluno ")

plt.show()
kmeans=KMeans(n_clusters=4,init="k-means++",max_iter=100,n_init=10,random_state=0)

y=kmeans.fit_predict(X)
plt.scatter(X[y==0,0],X[y==0,1],s=100,c="blue",label="target people")

plt.scatter(X[y==1,0],X[y==1,1],s=100,c="green",label="neglect people")

plt.scatter(X[y==2,0],X[y==2,1],s=100,c="cyan",label="kids with no income")

plt.scatter(X[y==3,0],X[y==3,1],s=100,c="yellow",label="2nd prority people")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="red",label="centroid")

plt.title("clients clusters")

plt.xlabel("Age")

plt.ylabel("annual income")

plt.legend()

plt.show
X=customer_data.iloc[:,[2,4]].values
from sklearn.cluster import KMeans

cluno=[]

for i in range (1,11):

    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=100,n_init=10,random_state=0)

    kmeans.fit(X)

    cluno.append(kmeans.inertia_)

plt.plot(range(1,11),cluno)

plt.xlabel("no of clusters")

plt.ylabel("cluno ")

plt.show()
kmeans=KMeans(n_clusters=4,init="k-means++",max_iter=100,n_init=10,random_state=0)

y=kmeans.fit_predict(X)
plt.scatter(X[y==0,0],X[y==0,1],s=100,c="blue",label="normal people")

plt.scatter(X[y==1,0],X[y==1,1],s=100,c="green",label="target people")

plt.scatter(X[y==2,0],X[y==2,1],s=100,c="cyan",label="neglect people")

plt.scatter(X[y==3,0],X[y==3,1],s=100,c="yellow",label="special people")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="red",label="centroid")

plt.title("clients clusters")

plt.xlabel("Age")

plt.ylabel("Spending Score")

plt.legend()

plt.show
X3 = customer_data[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X3)

    inertia.append(algorithm.inertia_)
plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , inertia , 'o')

plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X3)

labels3=algorithm.labels_

centroids3=algorithm.cluster_centers_
import plotly as py

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected = True)

customer_data['label3'] =  labels3

trace1 = go.Scatter3d(

    x= customer_data['Age'],

    y= customer_data['Spending Score (1-100)'],

    z= customer_data['Annual Income (k$)'],

    mode='markers',

     marker=dict(

        color = customer_data['label3'], 

        size= 20,

        line=dict(

            color= customer_data['label3'],

            width= 12

        ),

        opacity=0.8

     )

)

data = [trace1]

layout = go.Layout(

#     margin=dict(

#         l=0,

#         r=0,

#         b=0,

#         t=0

#     )

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)

plt.show()