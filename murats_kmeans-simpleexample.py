#importing necessary libraries



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#we can create own dataset. (gaussian distributation)



x1 = np.random.normal(25,5,1000)

y1 = np.random.normal(25,5,1000)



x2 = np.random.normal(55,5,1000)

y2 = np.random.normal(60,5,1000)



x3 = np.random.normal(55,5,1000)

y3 = np.random.normal(15,5,1000)
#almost dataset is ready.

x = np.concatenate((x1,x2,x3), axis = 0)

y = np.concatenate((y1,y2,y3), axis = 0)
dictionary = {"x":x, "y":y}
dictionary
df = pd.DataFrame(dictionary)

#dataset is ready, it called df.
plt.scatter(x1,y1)

plt.scatter(x2,y2)

plt.scatter(x3,y3)

plt.show()

#Unsupervised learning, it does'nt know labels.
plt.scatter(x,y)

plt.show()

#Yes, this way.
from sklearn.cluster import KMeans

WCSS = [] #within clusters sum of squares
#hey, i have a problem, how can i know optimum k value?

#sure, i prefer elbow rules.

for k in range(1,15):

    model = KMeans(n_clusters = k)

    model.fit(df)

    WCSS.append(model.inertia_)
#can you see elbow point? i guess it is 3.

plt.plot(range(1,15), WCSS)
#yes we can do it, anymore.

model = KMeans(n_clusters = 3)

clusters = model.fit_predict(df)
# we can create new columns, labels!

df["labels"] = clusters
plt.scatter(df.x[df.labels == 0], df.y[df.labels == 0], color="red")

plt.scatter(df.x[df.labels == 1], df.y[df.labels == 1], color="green")

plt.scatter(df.x[df.labels == 2], df.y[df.labels == 2], color="blue")

plt.show()

#did you see, it is perfect example of unsupervised learning.
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color = "black")
#if you wanna say that "where is the cluster centers ???" here you go.

plt.scatter(df.x[df.labels == 0], df.y[df.labels == 0], color="red")

plt.scatter(df.x[df.labels == 1], df.y[df.labels == 1], color="green")

plt.scatter(df.x[df.labels == 2], df.y[df.labels == 2], color="blue")

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color = "black")

plt.show()
#if you have any question or suggestion, i will be happy to hear it.