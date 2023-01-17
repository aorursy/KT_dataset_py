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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

iris = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
iris.head()
iris['species'].value_counts()
iris_df = iris.drop(columns = 'species')

iris_df.head()
iris_df.shape
iris_df.info()
iris_df.describe()
#Correlation Matrix

iris_df.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap("coolwarm"), axis = 1)
sns.pairplot(iris_df)
from sklearn.cluster import KMeans
#Create the model and use the default number of clusters

kmeans = KMeans(random_state = 42)

kmeans.fit(iris_df)
#Save the centroids value

centroids = kmeans.cluster_centers_
#Make the prediction

y_kmeans = kmeans.predict(iris_df)
#Elbow Method

list_clusters = list(range(1,9)) #until the kmeans default clusters number

inertia = []

for number_clusters in list_clusters:

    kmeans = KMeans(n_clusters = number_clusters, random_state = 42)

    kmeans.fit(iris_df)

    inertia.append(kmeans.inertia_)



#Plot the elbow chart

plt.plot(list_clusters, inertia, marker = "+", linestyle = 'solid', mec = 'red' )

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.show()
#Elbow Method STARTING WITH 2 CLUSTERS

list_clusters = list(range(2,9)) #until the kmeans default clusters number

inertia = []

for number_clusters in list_clusters:

    kmeans = KMeans(n_clusters = number_clusters, random_state = 42)

    kmeans.fit(iris_df)

    inertia.append(kmeans.inertia_)



#Plot the elbow chart

plt.plot(list_clusters, inertia, marker = "+", linestyle = 'solid', mec = 'red' )

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.show()
# Straight line between 2 clusters and 8 clusters

x0, y0 = 2, inertia[0]

x1, y1 = 8, inertia[-1]

X = [x0, x1]

Y = [y0, y1]

#Plot the elbow chart

plt.plot(list_clusters, inertia, marker = "+", linestyle = 'solid', mec = 'red' )

plt.plot(X, Y)

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.legend(['Elbow method', 'Straight line between clusters'])

plt.show()
inertia
list_clusters
#Make the distance function and decide the best number of clusters

from math import sqrt

def number_of_clusters(list_clusters, inertia):

    distances = []

    dictionary = {}

    x0, y0 = list_clusters[0], inertia[0]

    x1, y1 = list_clusters[-1], inertia[-1]

    i = 0

    for i in range(len(list_clusters)):

        x = list_clusters[i]

        y = inertia[i]

        a = abs((y1-y0)*x - (x1 - x0)*y + (x1*y0) - (y1*x0))

        b = sqrt((y1-y0)**2 + (x1-x0)**2)

        distances.append(a/b)

        dictionary = dict(zip(distances, list_clusters))

        i +=1     

    return print('Optimal number of clusters is:', dictionary.get(max(dictionary.keys())), "\nDistances between the point and the straight line: ", distances)
number_of_clusters(list_clusters, inertia)
#Create the model again but with n_clusters = 4 and adding a new column to the DataFrame

kmeans = KMeans(n_clusters = 4, random_state = 42)

kmeans.fit(iris_df)

iris_df['clusters'] = kmeans.predict(iris_df)
iris_df.head()
iris_df['clusters'].value_counts()
sns.pairplot(iris_df, hue = 'clusters')
#cluster chart of the  petal_length and the petal_width

plt.subplots(figsize = (12,7))

plt.subplot(1,2,1)

plt.title('K-means result')

plt.scatter(iris_df['petal_length'], iris_df['petal_width'], c = iris_df['clusters'], cmap = 'winter')

plt.xlabel('petal_length')

plt.ylabel('petal_width')



plt.subplot(1,2,2)

plt.title('Real classification')

plt.scatter(iris['petal_length'], iris['petal_width'], c = iris['species'].replace(iris['species'].value_counts().index, [0,1,2]), cmap = 'winter')

plt.xlabel('petal_length')

plt.ylabel('petal_width')