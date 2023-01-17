# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import sklearn.datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn import svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
iris = sklearn.datasets.load_iris()



data = iris.data[iris.target != 2,:2]

target = iris.target[iris.target != 2]

target = target*2 - 1
plt.scatter(data[:50,0],data[:50,1], label = "Setosa")

plt.scatter(data[50:,0],data[50:,1], label = "Versicolor")

plt.legend()

lda = LinearDiscriminantAnalysis(store_covariance=True)

clf = lda.fit(data, target)



plt.scatter(data[:50,0],data[:50,1], label = "Setosa")

plt.scatter(data[50:,0],data[50:,1], label = "Versicolor")

plt.legend()



# affichage de l'hyperplan séparateur

nx, ny = 200, 100

x_min, x_max = plt.xlim()

y_min, y_max = plt.ylim()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),

                   np.linspace(y_min, y_max, ny))

Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])

Z = Z[:, 1].reshape(xx.shape)

plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='black')



# affichage des moyennes empiriques

plt.scatter(clf.means_[0,0], clf.means_[0,1], marker="*", c="gold", s=500)

plt.scatter(clf.means_[1,0], clf.means_[1,1], marker="*", c="gold", s=500)

plt.xlabel("Longueur des sépales")

plt.ylabel("Largeur des sépales")
mean1 = [3, 0]

cov1 = [[2, 1.9], [1.9, 2]]  # diagonal covariance

x1, y1 = np.random.multivariate_normal(mean1, cov1, 50).T

mean2 = [0.5, 2]

cov2 = [[1, -0.9], [-0.9, 1]]   # diagonal covariance

x2, y2 = np.random.multivariate_normal(mean2, cov2, 50).T



data2 = np.zeros([100,2])



data2[:50,0] = x1

data2[:50,1] = y1

data2[50:,0] = x2

data2[50:,1] = y2
lda = LinearDiscriminantAnalysis(store_covariance=True)

clf = lda.fit(data2, target)



plt.scatter(data2[:50,0],data2[:50,1])

plt.scatter(data2[50:,0],data2[50:,1])



# affichage de l'hyperplan séparateur

nx, ny = 200, 100

x_min, x_max = plt.xlim()

y_min, y_max = plt.ylim()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),

                   np.linspace(y_min, y_max, ny))

Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])

Z = Z[:, 1].reshape(xx.shape)

plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='black')



# affichage des moyennes empiriques

plt.scatter(clf.means_[0,0], clf.means_[0,1], marker="*", c="gold", s=500)

plt.scatter(clf.means_[1,0], clf.means_[1,1], marker="*", c="gold", s=500)



logreg = LogisticRegression(solver = "lbfgs", C=1e42)

clf2 = logreg.fit(data2, target)
plt.scatter(data2[:50,0],data2[:50,1])

plt.scatter(data2[50:,0],data2[50:,1])



# affichage de l'hyperplan séparateur

nx, ny = 200, 100

x_min, x_max = plt.xlim()

y_min, y_max = plt.ylim()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),

                   np.linspace(y_min, y_max, ny))

Z = logreg.predict_proba(np.c_[xx.ravel(), yy.ravel()])

Z = Z[:, 1].reshape(xx.shape)

plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='black')



# affichage des moyennes empiriques

plt.scatter(clf.means_[0,0], clf.means_[0,1], marker="*", c="gold", s=500)

plt.scatter(clf.means_[1,0], clf.means_[1,1], marker="*", c="gold", s=500)
from sklearn import datasets

def make_meshgrid(x, y, h=.02):

    x_min, x_max = x.min() - 1, x.max() + 1

    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    return xx, yy





def plot_contours(ax, clf, xx, yy, **params):



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, **params)

    return out







iris = datasets.load_iris()



X = data

y = target



models = (svm.SVC(kernel='linear', C=C),

          svm.SVC(kernel='poly', degree=2, C=C),

          svm.SVC(kernel='rbf', gamma=0.7, C=C),

          svm.SVC(kernel='poly', degree=10, C=C))

models = (clf.fit(X, y) for clf in models)





titles = ('Noyau linéaire',

          'Noyau polynomial de degré 2',

          'Noyau RBF',

          'Noyau polynomial de degré 10')



fig, sub = plt.subplots(2, 2,figsize=(8,8))

plt.subplots_adjust(wspace=0.4, hspace=0.3)



X0, X1 = X[:, 0], X[:, 1]

xx, yy = make_meshgrid(X0, X1)



for clf, title, ax in zip(models, titles, sub.flatten()):

    plot_contours(ax, clf, xx, yy,

                  cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())

    ax.set_ylim(yy.min(), yy.max())

    ax.set_xlabel('Longueur des sépales')

    ax.set_ylabel('Largeur des sépales')

    ax.set_xticks(())

    ax.set_yticks(())

    ax.set_title(title)

    



plt.show()
