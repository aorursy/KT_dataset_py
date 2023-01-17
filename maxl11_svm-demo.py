import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

from sklearn import svm, metrics
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
samples = 500

train_prop = 0.8



# Make data

x, y = make_circles(n_samples=samples, noise=0.05, random_state=123)



# Plot

df = pd.DataFrame(dict(x=x[:, 0], y=x[:, 1], label=y))



groups = df.groupby('label')



fig, ax = plt.subplots()

ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

for name, group in groups:

    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)

ax.legend()

plt.show()
# Minmax scale



x = (x-x.min())/(x.max()-x.min())
# Linear

C = 1.0  # SVM regularization parameter

models = svm.SVC(kernel='linear', C=C)

models.fit(x, y)



# title for the plots

titles = ('SVC with linear kernel')



# Set-up 2x2 grid for plotting.

fig, sub = plt.subplots()

plt.subplots_adjust(wspace=0.4, hspace=0.4)



X0, X1 = x[:, 0], x[:, 1]

xx, yy = make_meshgrid(X0, X1)



plot_contours(sub, models, xx, yy,

              cmap=plt.cm.coolwarm, alpha=0.8)

sub.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

sub.set_xlim(-0.25, 1.25)

sub.set_ylim(-0.25, 1.25)

sub.set_xlabel('X')

sub.set_ylabel('Y')

sub.set_title(titles)



plt.show()
# Poly 3

C = 1.0  # SVM regularization parameter

models = svm.SVC(kernel='poly', degree=3, C=C, gamma='auto')

models.fit(x, y)



# title for the plots

titles = ('SVC with 3rd degree polynomial kernel')



# Set-up 2x2 grid for plotting.

fig, sub = plt.subplots()

plt.subplots_adjust(wspace=0.4, hspace=0.4)



X0, X1 = x[:, 0], x[:, 1]

xx, yy = make_meshgrid(X0, X1)



plot_contours(sub, models, xx, yy,

              cmap=plt.cm.coolwarm, alpha=0.8)

sub.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

sub.set_xlim(-0.25, 1.25)

sub.set_ylim(-0.25, 1.25)

sub.set_xlabel('X')

sub.set_ylabel('Y')

sub.set_title(titles)



plt.show()
# RBF

C = 1.0  # SVM regularization parameter

models = svm.SVC(kernel='rbf', gamma=0.7, C=C)

models.fit(x, y)



# title for the plots

titles = ('SVC with RBF kernel')



# Set-up 2x2 grid for plotting.

fig, sub = plt.subplots()

plt.subplots_adjust(wspace=0.4, hspace=0.4)



X0, X1 = x[:, 0], x[:, 1]

xx, yy = make_meshgrid(X0, X1)



plot_contours(sub, models, xx, yy,

              cmap=plt.cm.coolwarm, alpha=0.8)

sub.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

sub.set_xlim(-0.25, 1.25)

sub.set_ylim(-0.25, 1.25)

sub.set_xlabel('X')

sub.set_ylabel('Y')

sub.set_title(titles)



plt.show()