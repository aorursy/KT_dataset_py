# Lets plot a scatter chart, which shall help us in guessing the no. of clusters we may need



import matplotlib.pyplot as plt

x = [6.3, 5.8, 7.1, 5.1, 4.9, 4.7, 4.6, 7, 6.4, 6.9, 5.5]

y = [3.3, 2.7, 3, 3.5, 3, 3.2, 3.1, 3.2, 3.2, 3.1, 2.3]

plt.scatter(x,y)

plt.show()
import matplotlib.pyplot as plt

x = [6.3, 5.8, 7.1, 5.1, 4.9, 4.7, 4.6, 7, 6.4, 6.9, 5.5]

y = [3.3, 2.7, 3, 3.5, 3, 3.2, 3.1, 3.2, 3.2, 3.1, 2.3]



cx = [6.3, 5.1, 7]

cy = [3.3, 3.5, 3.2]



plt.scatter(x,y)

plt.scatter(cx,cy,label='Centroids',color='red',marker='1')



plt.legend()

plt.show()
import matplotlib.pyplot as plt

x = [6.3, 5.8, 7.1, 5.1, 4.9, 4.7, 4.6, 7, 6.4, 6.9, 5.5]

y = [3.3, 2.7, 3, 3.5, 3, 3.2, 3.1, 3.2, 3.2, 3.1, 2.3]



cx = [6.35, 5.1, 7]

cy = [3.25, 2.97, 3.1]



plt.scatter(x,y)

plt.scatter(cx,cy,label='New Centroids',color='red',marker='1')



plt.legend()

plt.show()
x1 = [6.3, 6.4]

y1 = [3.3, 3.2]

x2 = [5.8, 5.1, 4.9, 4.7, 4.6, 5.5]

y2 = [2.7, 3.5, 3, 3.2, 3.1, 2.3]

x3 = [7.1, 7, 6.9]

y3 = [3, 3.2, 3.1]

plt.scatter(x1,y1,color='r')

plt.scatter(x2,y2,color='b')

plt.scatter(x3,y3,color='g')

plt.scatter(cx,cy,label='Centroids',color='red',marker='1')

plt.legend()

plt.show()
x1 = [6.3, 5.8, 7.1]

y1 = [3.3, 2.7, 3]

x2 = [5.1, 4.9, 4.7, 4.6]

y2 = [3.5, 3, 3.2, 3.1]

x3 = [7, 6.4, 6.9, 5.5]

y3 = [3.2, 3.2, 3.1, 2.3]

plt.scatter(x1,y1,color='r')

plt.scatter(x2,y2,color='b')

plt.scatter(x3,y3,color='g')

plt.show()
#to read and format the Iris file

import pandas as pd 



#package to perform Kmeans algorithm

from sklearn.cluster import KMeans 



#For numerical functions

import numpy as np



#for graphs

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
#Read the Iris dataset



data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
#add headers to the file

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

data.columns = attributes

data.head()
# converting class names (string) to numbers which will aid us in plotting

data['class-num'] = data['class'].map( {'Iris-setosa': 0,'Iris-versicolor': 1,'Iris-virginica': 2} )
data.head()
#Define the features to be used for the algo i.e. remove column 'class' which is not required for the algo

X = data.drop(columns=["class","class-num"])



#Let's see the first five records in the file

X.head()
#Let's see the data characteristic

data.describe()
wcss = []

for i in range(1, 10):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 10), wcss)

plt.title('Elbow-Method using WCSS')

plt.xlabel('Number of Clusters (K)')

plt.ylabel('Within-Cluster-Sum-of-Squares')

plt.show()
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

kmeans.fit(X)
inertia = kmeans.predict(X)
X["Clusters"] = inertia

X.head()
fig = plt.figure()

ax = plt.axes(projection="3d")

ax.scatter3D(X['sepal_length'],X['sepal_width'],X['petal_length'],c=X['Clusters'],cmap='hsv')

plt.show()
fig = plt.figure()

ax = plt.axes(projection="3d")

ax.scatter3D(data['sepal_length'],data['sepal_width'],data['petal_length'],c=data['class-num'],cmap='hsv')

plt.show()