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
#iris virginica SVM



'''

need numpy 

import datasets and Pipeline, standard scalar, LinearSVC



1. load iris dataset

get X and y

make pipeline scales and svc hingloss loss function



fit

predict

'''
import numpy as np

from sklearn import datasets

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC



import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
iris = datasets.load_iris()

iris.keys()
X,y = make_moons(n_samples=100, noise=0.15)
def plot_dataset(X, y, axes):

    plt.plot(X[:,0][y == 1], X[:,1][y==1], "bs")

    plt.plot(X[:,0][y == 0], X[:,1][y==0], "g^")

    plt.axis(axes)

    plt.grid(True, which="both")

    plt.xlabel(r"$X_1$", fontsize=20)

    plt.ylabel(r"$X_2$", fontsize=20)



plot_dataset(X,y, [-1.5, 2.5, -1.0, 1.5])
from sklearn.preprocessing import PolynomialFeatures



pipeline = Pipeline([('poly_features', PolynomialFeatures(degree=3)),

                     ('std_scaler', StandardScaler()),

                     ('linear_svc', LinearSVC(C=10, loss='hinge'))])
pipeline.fit(X,y)
def plot_predictions(clf, axes):

    x0s = np.linspace(axes[0], axes[1], 100)

    x1s = np.linspace(axes[2], axes[3], 100)

    x0, x1 = np.meshgrid(x0s, x1s)

    X = np.c_[x0.ravel(), x1.ravel()]

    y_pred = clf.predict(X).reshape(x0.shape)

    y_decision = clf.decision_function(X).reshape(x0.shape)

    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)

    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
plot_predictions(pipeline, [-1.5, 2.5, -1, 1.5])

plot_dataset(X,y, [-1.5, 2.5, -1.0, 1.5])