# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy import sparse

from sklearn import datasets

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

random_state = 0
class KMeansFeaturizer(object):

    

    def __init__(self, n_clusters=8, random_state=None):

        self.n_clusters = n_clusters

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

        self.encoder = OneHotEncoder(categories=[np.arange(n_clusters)], sparse=True)

        

    def fit(self, X, y=None):

        self.kmeans.fit(X)

        return self

        

    def transform(self, X):

        labels = self.kmeans.predict(X)

        return sparse.hstack([X, self.encoder.fit_transform(np.c_[labels])])
def plot_decision_boundary(model, X, y):

    _min, _max = X.min(axis=0), X.max(axis=0)

    xx, yy = np.meshgrid(

        np.linspace(_min[0], _max[0], 50),

        np.linspace(_min[1], _max[1], 50)

    )

    X_test = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict_proba(X_test)[:, 1]

    Z = Z.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y)

    plt.contour(xx, yy, Z, levels=[0.5], colors='green')
X, y = datasets.make_moons(n_samples=500, noise=0.1, random_state=random_state)
# Let's try ordinal Logistic Regression.

model = LogisticRegression(solver='lbfgs', random_state=random_state)

model.fit(X, y)
plot_decision_boundary(model, X, y)
# As you can see that, Logistic Regression cannot express nonlinear decision boundary.

# I introduce (Deicision tree) featurizer to handle not linearly separable data.



model = Pipeline([

    ('featurizer', KMeansFeaturizer(n_clusters=16)),

    ('classifier', LogisticRegression(solver='lbfgs'))

])

model.fit(X, y)
plot_decision_boundary(model, X, y)
# Now we can handle not linearly separable data.

# This is just an example of featirizations. 