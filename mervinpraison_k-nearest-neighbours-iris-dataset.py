import matplotlib

#matplotlib.use('GTKAgg')



import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets



# import some data to play with

iris = datasets.load_iris()



# take the first two features

# choosing first 2 columns sepal length and sepal width

X = iris.data[:, :2]

y = iris.target

h = .02  # step size in the mesh



# Calculate min, max and limits

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



# Put the result into a color plot

plt.figure()

plt.scatter(X[:, 0], X[:, 1])

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.title("Data points")

plt.show()
import matplotlib



import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets



n_neighbors = 6



# import some data to play with

iris = datasets.load_iris()



# prepare data

# choosing first 2 columns sepal length and sepal width

X = iris.data[:, :2]

y = iris.target

h = .02



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])



# we create an instance of Neighbours Classifier and fit the data.

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

clf.fit(X, y)



# calculate min, max and limits

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

np.arange(y_min, y_max, h))



# predict class using data and kNN classifier

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

Z = Z.reshape(xx.shape)



plt.figure()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())



plt.figure()

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)



# Plot also the training points

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.title("3-Class classification (k = %i)" % (n_neighbors))

plt.show()
import matplotlib



import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets



n_neighbors = 6



# import some data to play with

iris = datasets.load_iris()



# prepare data

# choosing 3rd and 4th columns petal length and petal width

X = iris.data[:, [2,3]]

y = iris.target

h = .02



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])



# we create an instance of Neighbours Classifier and fit the data.

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

clf.fit(X, y)



# calculate min, max and limits

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

np.arange(y_min, y_max, h))



# predict class using data and kNN classifier

print(np.c_[xx.ravel(), yy.ravel()])



Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

print(Z)



# Put the result into a color plot

Z = Z.reshape(xx.shape)

print(Z)



plt.figure()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)



plt.figure()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())





plt.figure()

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)



# Plot also the training points

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.title("3-Class classification (k = %i)" % (n_neighbors))

plt.show()
import numpy as np

from sklearn import neighbors, datasets

from sklearn import preprocessing



n_neighbors = 6



# import some data to play with

iris = datasets.load_iris()



# prepare data

# choosing first 2 columns sepal length and sepal width

X = iris.data[:, :2]

y = iris.target

h = .02



# we create an instance of Neighbours Classifier and fit the data.

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

clf.fit(X, y)



# make prediction

#sl = raw_input('Enter sepal length (cm): ')

#sw = raw_input('Enter sepal width (cm): ')



sl = 6.4

sw = 2.8

dataClass = clf.predict([[sl,sw]])

print('Prediction: '),



if dataClass == 0:

    print('Iris Setosa')

elif dataClass == 1:

    print('Iris Versicolour')

else:

    print('Iris Virginica')