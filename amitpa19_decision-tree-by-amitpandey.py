# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier



iris = load_iris()

X = iris.data[:, 2:] # petal length and width

y = iris.target



tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)

tree_clf.fit(X, y)
from matplotlib.colors import ListedColormap



def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):

    x1s = np.linspace(axes[0], axes[1], 100)

    x2s = np.linspace(axes[2], axes[3], 100)

    x1, x2 = np.meshgrid(x1s, x2s)

    X_new = np.c_[x1.ravel(), x2.ravel()]

    y_pred = clf.predict(X_new).reshape(x1.shape)

    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)

    if not iris:

        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])

        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)

    if plot_training:

        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")

        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")

        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")

        plt.axis(axes)

    if iris:

        plt.xlabel("Petal length", fontsize=14)

        plt.ylabel("Petal width", fontsize=14)

    else:

        plt.xlabel(r"$x_1$", fontsize=18)

        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

    if legend:

        plt.legend(loc="lower right", fontsize=14)



plt.figure(figsize=(8, 4))

plot_decision_boundary(tree_clf, X, y)

plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)

plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)

plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)

plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)

plt.text(1.40, 1.0, "Depth=0", fontsize=15)

plt.text(3.2, 1.80, "Depth=1", fontsize=13)

plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)



plt.show()
tree_clf.predict_proba([[5, 1.5]])
X[(X[:, 1]==X[:, 1][y==1].max()) & (y==1)] # widest Iris-Versicolor flower
not_widest_versicolor = (X[:, 1]!=1.8) | (y==2)

X_tweaked = X[not_widest_versicolor]

y_tweaked = y[not_widest_versicolor]



tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)

tree_clf_tweaked.fit(X_tweaked, y_tweaked)
plt.figure(figsize=(8, 4))

plot_decision_boundary(tree_clf_tweaked, X_tweaked, y_tweaked, legend=False)

plt.plot([0, 7.5], [0.8, 0.8], "k-", linewidth=2)

plt.plot([0, 7.5], [1.75, 1.75], "k--", linewidth=2)

plt.text(1.0, 0.9, "Depth=0", fontsize=15)

plt.text(1.0, 1.80, "Depth=1", fontsize=13)



plt.show()
from sklearn.datasets import make_moons

Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)



deep_tree_clf1 = DecisionTreeClassifier(random_state=42)

deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)

deep_tree_clf1.fit(Xm, ym)

deep_tree_clf2.fit(Xm, ym)



plt.figure(figsize=(11, 4))

plt.subplot(121)

plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)

plt.title("No restrictions", fontsize=16)

plt.subplot(122)

plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)

plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)



plt.show()
angle = np.pi / 180 * 20

rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

Xr = X.dot(rotation_matrix)



tree_clf_r = DecisionTreeClassifier(random_state=42)

tree_clf_r.fit(Xr, y)



plt.figure(figsize=(8, 3))

plot_decision_boundary(tree_clf_r, Xr, y, axes=[0.5, 7.5, -1.0, 1], iris=False)



plt.show()
np.random.seed(6)

Xs = np.random.rand(100, 2) - 0.5

ys = (Xs[:, 0] > 0).astype(np.float32) * 2



angle = np.pi / 4

rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

Xsr = Xs.dot(rotation_matrix)



tree_clf_s = DecisionTreeClassifier(random_state=42)

tree_clf_s.fit(Xs, ys)

tree_clf_sr = DecisionTreeClassifier(random_state=42)

tree_clf_sr.fit(Xsr, ys)



plt.figure(figsize=(11, 4))

plt.subplot(121)

plot_decision_boundary(tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)

plt.subplot(122)

plot_decision_boundary(tree_clf_sr, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)



plt.show()