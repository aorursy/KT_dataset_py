import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib



import os

import warnings

warnings.filterwarnings('ignore')
dfs = []



for file in sorted(os.listdir("../input/toysets")):

    dfs.append(pd.read_csv("../input/toysets/" + file, names=['x1', 'x2', 'y']))
def plot_boundaries(X, y, clf, plt):

    """FOR STANDALONE PLOTS

    

    X, y: TEST/CV data

    clf: sklearn classifier

    """

    plt.close()

    

    # Plot the decision boundary. For that, we will assign a color to each

    # point in the mesh [x_min, x_max]x[y_min, y_max].

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    h = .02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.figure(1, figsize=(4, 3))

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)



    # plot actual data

    plt.scatter(X[:,0], X[:, 1], c=y.ravel(),

            edgecolors="k", cmap=plt.cm.Paired)

    

    plt.axis('off')

    

    plt.show()
def plot_boundaries_on_ax(X, y, clf, ax):

    """FOR SUBPLOTS

    

    X, y: TEST/CV data

    clf: sklearn classifier

    """

    

    # Plot the decision boundary. For that, we will assign a color to each

    # point in the mesh [x_min, x_max]x[y_min, y_max].

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    h = .02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)



    # plot actual data

    ax.scatter(X[:,0], X[:, 1], c=y.ravel(),

            edgecolors="k", cmap=plt.cm.Paired)

    

    ax.axis('off')
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import math
ks = [1, 5, 15, 30, 45]





for df in dfs:

    xs = df.drop(['y'], axis=1)

    ys = df[['y']]





    # config subplot

    # ---------------

    plt.close() # open subplot



    cols = 5

    rows = math.ceil(len(ks) / cols)

    fig, axarr = plt.subplots(rows,cols, figsize=(5*cols, 3*rows))

    axarr = axarr.flatten()



    # plot all ks for a specific dataset

    for idx, k in enumerate(ks):



        # build model

        # -----------

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(xs, ys) # train and plot all points

        # plot

        # ----------

        plot_boundaries_on_ax(np.array(xs), np.array(ys), knn, axarr[idx])



    plt.show() # show subplot