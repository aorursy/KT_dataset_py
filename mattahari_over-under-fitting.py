# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score





def true_fun(X):

    return np.cos(1.5 * np.pi * X)



np.random.seed(0)



n_samples = 30

degrees = [1, 4, 15]



X = np.sort(np.random.rand(n_samples))

y = true_fun(X) + np.random.randn(n_samples) * 0.1



def plot_example(i):

    ax = plt.subplot(1, len(degrees), i + 1)

    plt.setp(ax, xticks=(), yticks=())



    polynomial_features = PolynomialFeatures(degree=degrees[i],

                                             include_bias=False)

    linear_regression = LinearRegression()

    pipeline = Pipeline([("polynomial_features", polynomial_features),

                         ("linear_regression", linear_regression)])

    pipeline.fit(X[:, np.newaxis], y)



    # Evaluate the models using crossvalidation

    scores = cross_val_score(pipeline, X[:, np.newaxis], y,

                             scoring="neg_mean_squared_error", cv=10)



    X_test = np.linspace(0, 1, 100)

    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")

    plt.plot(X_test, true_fun(X_test), label="True function")

    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")

    plt.xlabel("x")

    plt.ylabel("y")

    plt.xlim((0, 1))

    plt.ylim((-2, 2))

    plt.legend(loc="best")

    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(

        degrees[i], -scores.mean(), scores.std()))

    plt.show()
plot_example(0)
plt.close('all')

plt.figure(figsize=(20,10))

plot_example(2)
plt.close('all')

plt.figure(figsize=(20,10))

plot_example(1)