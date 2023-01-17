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



from mlxtend.plotting import plot_decision_regions

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets



from sklearn.svm import SVC

from sklearn.datasets.samples_generator import make_blobs, make_circles

from mpl_toolkits import mplot3d

from ipywidgets import interact, fixed

# Any results you write to the current directory are saved as output.

iris = datasets.load_iris()
X = iris.data[:, [2, 0]]

X = X[:100]

y = iris.target

y = y[:100]

svm = SVC(C=0.5, kernel='linear')

svm.fit(X, y)



fig = plt.figure(figsize=(16, 10))

# Plotting decision regions

plot_decision_regions(X, y, clf=svm, legend=2, zoom_factor=4.0)



# Adding axes annotations

plt.xlabel('petal length [cm]')

plt.ylabel('sepal length [cm]')

plt.title('SVM')

plt.show()
fig = plt.figure(figsize=(16, 10))

plt.scatter(x=[1,2,3,4,4,3.5,3.1,2.5,2.2, 4.9], y=[1,2,3,4,5,2,3,1,2,4], marker='o', s=100)

plt.scatter(x=[5.5, 6, 7, 8, 7,6.4,7.7,7.6, 5.4,6], y=[4,3,5,6,7,6,7,5,6,7.7], marker='+', s=100)

plt.plot([5.3,5], [0,8], 'ro-')
fig = plt.figure(figsize=(16, 10))

plt.scatter(x=[1,2,3,4,4,3.5,3.1,2.5,2.2, 4.9], y=[1,2,3,4,5,2,3,1,2,4], marker='o', s=100)

plt.scatter(x=[5.5, 6, 7, 8, 7,6.4,7.7,7.6, 5.4,6, 7.25], y=[4,3,5,6,7,6,7,5,6,7.7, 4.4], marker='+', s=100)

plt.plot([5.3,5], [0,8], 'ro-')

plt.plot([5.15, 7.2], [4,4.4], '>-g', color='green')

plt.text(4.95, 4, 'B', fontsize=22)

plt.text(7.3, 4.4, 'A', fontsize=22)

plt.plot([5.1, 7.2], [4+2,4.4+2], '>-g', color='blue')

plt.text(6, 5.6, '$W$', fontsize=22)

plt.text(6, 3.5, '$\gamma^{(i)}$', fontsize=22)
c = plt.Circle((0.5, 0.5), 0.2, fill=False)

fig, ax = plt.subplots(figsize=(16, 10))

ax.add_artist(c)

plt.text(0.45, 0.5, '$||w|| = 1$', fontsize=22)
y = [1 for i in range(2450)] + [-1 for i in range(2550)]

np.random.shuffle(y)

x = np.random.randint(1, 1000 + 1, size=5000)

w = -1

b = 1

gamma_hat = (y * (w * x + b))

fig, ax = plt.subplots(figsize=(16, 10))

plt.scatter(x, gamma_hat)
X, y = make_blobs(n_samples=50, centers=2,

                  random_state=0, cluster_std=0.60)



model = SVC(kernel='linear', C=1E10)

model.fit(X, y)



fig, ax = plt.subplots(figsize=(16, 10))

plt.scatter(X[:, 0], X[:, 1], c=y, s=50)



xlim = ax.get_xlim()

ylim = ax.get_ylim()



# create grid to evaluate model

x = np.linspace(xlim[0], xlim[1], 30)

y = np.linspace(ylim[0], ylim[1], 30)

Y, X = np.meshgrid(y, x)

xy = np.vstack([X.ravel(), Y.ravel()]).T

P = model.decision_function(xy).reshape(X.shape)



# plot decision boundary and margins

ax.contour(X, Y, P, colors='k',

           levels=[-1, 0, 1], alpha=0.5,

           linestyles=['--', '-', '--'])



# plot support vectors

ax.scatter(model.support_vectors_[:, 0],

           model.support_vectors_[:, 1],

           s=300, linewidth=1, facecolors='none');

ax.set_xlim(xlim)

ax.set_ylim(ylim)
fig, ax = plt.subplots(figsize=(16, 10))

def plot_svc_decision_function(model, ax=None, plot_support=True):

    """Plot the decision function for a 2D SVC"""

    if ax is None:

        ax = plt.gca()

    xlim = ax.get_xlim()

    ylim = ax.get_ylim()

    

    # create grid to evaluate model

    x = np.linspace(xlim[0], xlim[1], 30)

    y = np.linspace(ylim[0], ylim[1], 30)

    Y, X = np.meshgrid(y, x)

    xy = np.vstack([X.ravel(), Y.ravel()]).T

    P = model.decision_function(xy).reshape(X.shape)

    

    # plot decision boundary and margins

    ax.contour(X, Y, P, colors='k',

               levels=[-1, 0, 1], alpha=0.5,

               linestyles=['--', '-', '--'])

    

    # plot support vectors

    if plot_support:

        ax.scatter(model.support_vectors_[:, 0],

                   model.support_vectors_[:, 1],

                   s=300, linewidth=1, facecolors='none');

    ax.set_xlim(xlim)

    ax.set_ylim(ylim)



X, y = make_circles(100, factor=.1, noise=.1)



clf = SVC(kernel='linear').fit(X, y)



plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

plot_svc_decision_function(clf, plot_support=False)
fig, ax = plt.subplots(figsize=(16, 10))

r = np.exp(-(X ** 2).sum(1))



def plot_3D(elev=30, azim=30, X=X, y=y):

    ax = plt.subplot(projection='3d')

    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')

    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    ax.set_zlabel('r')



plot_3D(X=X, y=y)
fig, ax = plt.subplots(figsize=(16, 10))

clf = SVC(kernel='rbf', C=1E6, gamma='auto')

clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

plot_svc_decision_function(clf)

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],

            s=300, lw=1, facecolors='none')
fig, ax = plt.subplots(figsize=(16, 10))

clf = SVC(kernel='poly', degree = 2, C=1E6, gamma='auto')

clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

plot_svc_decision_function(clf)

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],

            s=300, lw=1, facecolors='none')
fig, axs = plt.subplots(figsize=(16, 50), nrows = 8)



for i in range(8):

    clf = SVC(kernel='poly', degree = i+3, C=1E6, gamma='auto')

    clf.fit(X, y)

    axs[i].set_title("Degree: " + str(i+3))

    axs[i].scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

    plot_svc_decision_function(clf, ax=axs[i])

    axs[i].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],

                s=300, lw=1, facecolors='none')