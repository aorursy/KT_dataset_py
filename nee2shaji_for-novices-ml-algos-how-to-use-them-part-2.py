import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets



# import some data to play with

iris = datasets.load_iris()



# we only take the first two features. We could avoid this ugly slicing by using a two-dim dataset

X = iris.data[:, :2]

y = iris.target



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) #for background light color

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  #for scatter points color



for k in [1, 5, 10]:

    # we create an instance of Neighbours Classifier and fit the data.

    clf = neighbors.KNeighborsClassifier(metric='euclidean', n_neighbors=k, weights='uniform')

    clf.fit(X, y)



    # Plot the decision boundary. For that, we will assign a color to each

    # point in the mesh [x_min, x_max]x[y_min, y_max].

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # np.meshgrid - Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields 

    # over N-D grids, given one-dimensional coordinate arrays x1, x2,…, xn

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.figure()

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)



    # Plot also the training points

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    plt.title("3-Class classification (k = %i, weights = 'uniform')" % (k))



plt.show()
import numpy as np

import matplotlib.pyplot as plt

from sklearn import neighbors



np.random.seed(0)

X = np.sort(5 * np.random.rand(40, 1), axis=0)

y = np.sin(X).ravel()



# Add noise to targets

y[::5] += 1 * (0.5 - np.random.rand(8))



# Fit regression model

n_neighbors = 5

weights = 'uniform'

knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)



# Now predict for values in T

T = np.linspace(0, 5, 500)[:, np.newaxis]

y_ = knn.fit(X, y).predict(T)



plt.scatter(X, y, c='k', label='data')

plt.plot(T, y_, c='g', label='prediction')

plt.legend()

plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))



plt.show()


import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm, datasets





def make_meshgrid(x, y, h=.02):

    """Create a mesh of points to plot in



    Parameters

    ----------

    x: data to base x-axis meshgrid on

    y: data to base y-axis meshgrid on

    h: stepsize for meshgrid, optional



    Returns

    -------

    xx, yy : ndarray

    """

    x_min, x_max = x.min() - 1, x.max() + 1

    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    return xx, yy





def plot_contours(ax, clf, xx, yy, **params):

    """Plot the decision boundaries for a classifier.



    Parameters

    ----------

    ax: matplotlib axes object

    clf: a classifier

    xx: meshgrid ndarray

    yy: meshgrid ndarray

    params: dictionary of params to pass to contourf, optional

    """

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, **params)

    return out





# import some data to play with

iris = datasets.load_iris()

# Take the first two features. We could avoid this by using a two-dim dataset

X = iris.data[:, :2]

y = iris.target



# we create an instance of SVM and fit out data. We do not scale our

# data since we want to plot the support vectors

C = 1.0  # SVM regularization parameter

models = (svm.SVC(kernel='linear', C=C),

          svm.LinearSVC(C=C, max_iter=10000),

          svm.SVC(kernel='rbf', gamma=0.7, C=C),

          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))

models = (clf.fit(X, y) for clf in models)



# title for the plots

titles = ('SVC with linear kernel',

          'LinearSVC (linear kernel)',

          'SVC with RBF kernel',

          'SVC with polynomial (degree 3) kernel')



# Set-up 2x2 grid for plotting.

fig, sub = plt.subplots(2, 2)

plt.subplots_adjust(wspace=0.4, hspace=0.4)



X0, X1 = X[:, 0], X[:, 1]

xx, yy = make_meshgrid(X0, X1)



for clf, title, ax in zip(models, titles, sub.flatten()):

    plot_contours(ax, clf, xx, yy,

                  cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())

    ax.set_ylim(yy.min(), yy.max())

    ax.set_xlabel('Sepal length')

    ax.set_ylabel('Sepal width')

    ax.set_xticks(())

    ax.set_yticks(())

    ax.set_title(title)



plt.show()
from sklearn.datasets import load_iris

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 8, random_state=0)

iris = load_iris()

cross_val_score(clf, iris.data, iris.target, cv=10)