import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
iris = pd.read_csv("../input/Iris.csv")

iris.shape
iris.head()
X = iris.iloc[:, 1:5].values

y = pd.Categorical(iris['Species']).codes
from sklearn.cluster import KMeans

estimators = {'k_means_iris_3': KMeans(n_clusters=3),

              'k_means_iris_8': KMeans(n_clusters=8),

              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,

                                              init='random')}
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



fignum = 1

for name, est in estimators.items():

    fig = plt.figure(fignum, figsize=(8, 6))

    plt.clf()

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)



    plt.cla()

    est.fit(X)

    labels = est.labels_



    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))



    ax.w_xaxis.set_ticklabels([])

    ax.w_yaxis.set_ticklabels([])

    ax.w_zaxis.set_ticklabels([])

    ax.set_xlabel('Petal width')

    ax.set_ylabel('Sepal length')

    ax.set_zlabel('Petal length')

    ax.set_title(name, loc='left', fontsize=15)

    fignum = fignum + 1
# Plot the ground truth

fig = plt.figure(fignum, figsize=(8, 6))

plt.clf()

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)



plt.cla()

for name, label in [('Setosa', 0),

                    ('Versicolour', 1),

                    ('Virginica', 2)]:

    ax.text3D(X[y == label, 3].mean(),

              X[y == label, 0].mean() + 1.5,

              X[y == label, 2].mean(), name,

              horizontalalignment='center',

              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results

y = np.choose(y, [1, 2, 0]).astype(np.float)

ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)



ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('Petal width')

ax.set_ylabel('Sepal length')

ax.set_zlabel('Petal length')

plt.show()