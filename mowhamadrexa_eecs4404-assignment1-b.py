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



# Any results you write to the current directory are saved as output.
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_diabetes

from sklearn import linear_model

from sklearn.metrics import mean_squared_error





def fit_LinRegr(X_train, y_train):

    # adding x0 = 1 to every input data X

    X = np.insert(X_train, 0, 1, axis=1)

    # Defining Y

    Y = y_train



    # w = (X^T . X)^(-1)  . X^T . Y

    w = (X.transpose().dot(X))

    w = np.linalg.inv(w).dot(X.transpose()).dot(Y)



    return w





def mse(X_train, y_train, w):

    # adding x0 = 1 to every input data X

    X = np.insert(X_train, 0, 1, axis=1)

    # Defining Y

    Y = y_train

    # Defining average squared error = 0

    error = 0



    # iterating through all of input

    for index, i in enumerate(X):

        # error = error + || Y - X.w ||^2

        error += (pred(i, w) - Y[index]) ** 2



    # error = 1/N (error), N: # of input data

    error /= len(X)

    return error





def pred(X_train, w):

    # returning X . w

    return X_train @ w





def test_SciKit(X_train, X_test, Y_train, Y_test):

    # lnreg is an instance of LinearRegression

    lnreg = linear_model.LinearRegression()

    # Fitting data

    lnreg.fit(X_train, Y_train)

    # getting the prediction array, called Y_pred

    Y_pred = lnreg.predict(X_test)

    # returning the mean squared error of prediction

    return mean_squared_error(Y_test, Y_pred)





def testFn_Part2():

    X_train, y_train = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)



    # Testing Part 2a

    w = fit_LinRegr(X_train, y_train)

    e = mse(X_test, y_test, w)



    # Testing Part 2b

    scikit = test_SciKit(X_train, X_test, y_train, y_test)



    print("Mean squared error from Part 2a is\n", e)

    print("Mean squared error from Part 2b is\n", scikit)





testFn_Part2()



# How close is the performance of your implementation in comparison to the existing modules in the

# scikit-learn library?

# It's really close, like, really really close
