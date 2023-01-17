import numpy as np # linear algebra

from sklearn.datasets import load_iris

from sklearn.svm import SVC

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.
iris = load_iris()

X = iris.data[:, 0:2]

Y = iris.target
clf = SVC(kernel='linear')

clf.fit(X, Y)
x1_min = X[:, 0].min()-1

x2_min = X[:, 1].min()-1

x1_max = X[:, 0].max()+1

x2_max = X[:, 1].max()+1



h = (x1_max - x1_min)/100

x1_range = np.arange(x1_min, x1_max, h)

h = (x2_max - x2_min)/100

x2_range = np.arange(x2_min, x2_max, h)
xx, yy = np.meshgrid(x1_range, x2_range)

points = np.c_[xx.ravel(), yy.ravel()]
pred = clf.predict(points)
plt.contourf(xx, yy, np.reshape(pred, (100, 100)))

plt.scatter(X[:, 0], X[:, 1], c=Y)

plt.show()
clf = SVC(kernel='rbf')

clf.fit(X, Y)

pred = clf.predict(points)

plt.contourf(xx, yy, np.reshape(pred, (100, 100)))

plt.scatter(X[:, 0], X[:, 1], c=Y)

plt.show()