# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install skylearn





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output




# support for large, multi-dimensional arrays and matrices

import numpy as np

# plotting library for the Python programming language and its numerical mathematics extension NumPy

import matplotlib.pyplot as plt

from matplotlib import style

style.use("ggplot")

#for less number of datapoints KMeans for more than thousand of points 

#KMeansbatch can be used

from sklearn.cluster import KMeans







x= [1.5,2.8,2,6,5.5,6,8,5.3]

y= [3.2, 5, 2.2, 7, 6.9, 5,3.2,6.9]

plt.scatter(x,y)

plt.show()







#converting above co-ordinates into numpy list of list

X= np.array([[0.8,5],[2.5,4],[1, 1.8],[8,8],[6, 9.6],[7.5,10],[2, 3.6],[7.3,8.5]])



#assign number of clusters to machine

kmeans = KMeans(n_clusters=2)

kmeans.fit(X)




#clustering performed based on equal variance from centroid. 

centroids = kmeans.cluster_centers_

#semisupervised approach based on these label new data 

#points will be assigned label by algorithm.

labels = kmeans.labels_







print(centroids)

print(labels)



#assign number of clusters to machine

kmeans = KMeans(n_clusters=2)

kmeans.fit(X)
#converting above co-ordinates into numpy list of list

X= np.array([[0.5,7],[3.5,4],[1, 1.8],[8,8],[8, 9.6],[7.5,9],[4, 3.6],[7.3,8.5]])
#assign number of clusters to machine

kmeans = KMeans(n_clusters=2)

kmeans.fit(X)
#clustering performed based on equal variance from centroid. 

centroids = kmeans.cluster_centers_

#semisupervised approach based on these label new data 

#points will be assigned label by algorithm.

labels = kmeans.labels_
print(centroids)

print(labels)
#assign number of clusters to machine

kmeans = KMeans(n_clusters=2)

kmeans.fit(X)
#clustering performed based on equal variance from centroid. 

centroids = kmeans.cluster_centers_

#semisupervised approach based on these label new data 

#points will be assigned label by algorithm.

labels = kmeans.labels_
print(centroids)

print(labels)
colors = ["180","280"]
for i in range(len(X)):

    print("coordinate:", X[i], "label:", labels[i])

    plt.scatter(x,y, c=[colors[l_] for l_ in labels], label=labels)

    plt.scatter(centroids[:, 0],centroids[:, 1], c=[c for c in colors[:len(centroids)]], marker = "x", s=170, linewidths = 5, zorder = 10)

    plt.show()