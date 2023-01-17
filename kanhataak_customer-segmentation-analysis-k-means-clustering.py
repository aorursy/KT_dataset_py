import numpy as np # linear algebra

import pandas as pd # data processing, csv file



import matplotlib.pyplot as plt

import seaborn as sn



import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans



import warnings

import os



warnings.filterwarnings("ignore")

py.offline.init_notebook_mode(connected=True)



%matplotlib inline
df = pd.read_csv("../input/mall-customerscsv/Mall_Customers.csv")

df.head()
# Check the Descriptions of the datasets

df.describe()
df.tail()
# Know more about the data

df.info()
#  Check the null values in data using missingo

import missingno as mgo

mgo.matrix(df)
plt.figure(figsize=(8,5))

plt.scatter("Annual Income (k$)","Spending Score (1-100)", data= df,s=30, color="orange",alpha=0.8)

plt.xlabel("Annual Income")

plt.ylabel("Spendign Score")
plt.figure(figsize=(16,5))

sn.set(style = 'whitegrid')



plt.subplot(1,3,1)

sn.distplot(df['Age'])

plt.title('Distribution of Age', fontsize = 20)

plt.ylabel("Count")



plt.subplot(1,3,2)

sn.distplot(df['Annual Income (k$)'], color="red")

plt.title('Distribution of Annual Income', fontsize = 20)



plt.subplot(1,3,3)

sn.distplot(df['Spending Score (1-100)'], color="orange")

plt.title('Distribution of Spending Score', fontsize = 20)



plt.show()
count_classes = pd.value_counts(df['Gender'], sort = True)



count_classes.plot(kind = 'bar', rot=0,)



plt.title("Distribution of Gender")



plt.xticks(range(2))



plt.xlabel("Gender")



plt.ylabel("Frequency Count")
df_id = df.drop(['CustomerID'], axis=1)

df_id.hist(figsize = (15, 12))

plt.show()
plt.figure(figsize = (16,8))

sn.heatmap(df.corr(), cmap = 'Wistia', annot = True)

plt.title('Heatmap for the Data')

plt.show()
plt.figure(figsize=(20,8))

sn.countplot(df["Age"], palette="hsv")

plt.title("Distribution of each Age", fontsize=18)
df["Annual Income (k$)"].value_counts().plot(kind = 'bar',grid=False,figsize=(20,8), color="orange")

plt.title("Distribution of Anuual Income", fontsize=18)

plt.show()
df["Spending Score (1-100)"].value_counts().plot(kind = 'bar',figsize=(20,8), color="red")

plt.title("Distribution of Spending Score", fontsize=18)

plt.show()
# Clustering Analysis ---> Get the Annual Income and Spending Score

x= df.iloc[:,[3,4]].values
inertia=[]





for k in range(1,8):

    algo = KMeans(n_clusters = k,init="k-means++",max_iter=300,random_state=100)

    algo = algo.fit(x)

    inertia.append(algo.inertia_)
plt.figure(1 , figsize = (16 ,6))

plt.plot(np.arange(1 , 8) , inertia , 'o')

plt.plot(np.arange(1 , 8) , inertia , '-' )

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
algo = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = algo.fit_predict(x)



plt.figure(figsize=(8,5))

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'red', label = 'miser')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'green', label = 'target')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(algo.cluster_centers_[:,0], algo.cluster_centers_[:, 1], s = 200, c = 'blue' , label = 'centeroid',alpha=0.5)



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('Annual Income'), plt.ylabel('Spending Score')

plt.legend()

plt.grid()

plt.show()
x2 = df.drop(['CustomerID','Gender','Annual Income (k$)'],axis=1).values
x3=[]



for k in range(1,11):

    algo = KMeans(n_clusters = k)

    algo = algo.fit(x2)

    x3.append(algo.inertia_)
plt.figure(1 , figsize = (16 ,6))

plt.plot(np.arange(1 , 11) , x3 , 'o')

plt.plot(np.arange(1 , 11) , x3 , '-' )

plt.title("Choose the optimal K")

plt.xlabel('Number of Clusters')

plt.ylabel('Sum of squared distances/Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(x2)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_
h = 0.02

x_min, x_max = x2[:, 0].min() - 1, x2[:, 0].max() + 1

y_min, y_max = x2[:, 1].min() - 1, x2[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'green' , alpha = 0.5)

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')

plt.show()