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
import numpy as np

from sklearn import datasets

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

iris = datasets.load_iris()

X = iris["data"][:, (2,3)]

y = (iris["target"] == 2).astype(np.float64)



svm_clf = Pipeline([

    ("scaler", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge")),

                   ])

                   

svm_clf.fit(X, y)
svm_clf.predict([[5.5, 1.7]])
from sklearn.datasets import make_moons

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures



X, y = make_moons(n_samples=100, noise=0.15)

polynomial_svm_clf = Pipeline([

        ("poly_features", PolynomialFeatures(degree=3)),

        ("scaler", StandardScaler()),

        ("svm_clf", LinearSVC(C=10, loss="hinge"))

    ])

polynomial_svm_clf.fit(X, y)
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([

        ("scaler", StandardScaler()),

        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))

    ])

poly_kernel_svm_clf.fit(X, y)
rbf_kernel_svm_clf = Pipeline([

        ("scaler", StandardScaler()),

        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))

    ])

rbf_kernel_svm_clf.fit(X, y)
from sklearn.svm import LinearSVR



svm_reg = LinearSVR(epsilon=1.5)

svm_reg.fit(X, y)
from sklearn.svm import SVR



svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)

svm_poly_reg.fit(X, y)
