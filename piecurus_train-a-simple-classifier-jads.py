# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# load the iris dataset as an example

from sklearn.datasets import load_iris

iris = load_iris()



# Any results you write to the current directory are saved as output.
# store the feature matrix (X) and response vector (y)

X = iris.data

y = iris.target
# check the shapes of X and y

print(X.shape)

print(y.shape)
# examine the first 5 rows of the feature matrix (including the feature names)

pd.DataFrame(X, columns=iris.feature_names).head()
# examine the response vector

print(y)
# import the class

from sklearn.neighbors import KNeighborsClassifier



# instantiate the model (with the default parameters)

knn = KNeighborsClassifier()



# fit the model with data (occurs in-place)

knn.fit(X, y)
# predict the response for a new observation

knn.predict([[3, 5, 4, 2]])
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA



x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5



plt.figure(2, figsize=(8, 6))

plt.clf()



# Plot the training points

plt.scatter(X[:, 0], X[:, 1], s=100,c=y, cmap=plt.cm.CMRmap)

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')



plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.xticks(())

plt.yticks(())



# To getter a better understanding of interaction of the dimensions

# plot the first three PCA dimensions

fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], s=100,c=y,

           cmap=plt.cm.CMRmap)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])