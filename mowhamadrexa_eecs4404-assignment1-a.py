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

from sklearn.linear_model import Perceptron

from sklearn.metrics import confusion_matrix





def fit_perceptron(X_train, y_train):

    # defining w as weight vector and assigning zero values to it

    w = np.zeros((X_train.shape[1]) + 1)



    # adding x0 = 1 to all of our data inputs

    X_train = np.insert(X_train, 0, 1, axis=1)



    # we define epochs to keep track of number of times we go through

    # our input dataset

    epochs = 0



    # We define this dictionary to keep track of the average error and

    # W corresponding to that average error.

    # average error is Key, and w vector is Value

    # we initialize the error_w with greatest possible average error, 1

    # So we can make sure every other average error will be less that it

    error_w = {1: w}



    # last_round_min_error keeps the minimum value of last epoch

    last_round_min_error = 10





    while min(error_w) < last_round_min_error and epochs < 1001:

        # memorizing this round min error for the next condition to be used

        last_round_min_error = min(error_w)

        # increasing epoch by 1

        epochs += 1



        for index, data in enumerate(X_train):

            dot_product = w @ data

            # updating w based on the logic of algorithm

            if np.sign(dot_product) != np.sign(y_train[index]):

                w = w + (y_train[index] * data)

                # memorizing the average error and its correspondence w

                error_w[errorPer(X_train, y_train, w)] = w



    return error_w[min(error_w)]





def errorPer(X_train, y_train, w):

    y_pred = pred(X_train, w)



    not_equal = 0

    for index, data in enumerate(X_train):

        # if the prediction is not equal to actual value, error += 1

        if np.sign(y_pred[index]) != np.sign(y_train[index]):

            not_equal += 1



    # error / N , N: # of input data

    return not_equal / len(y_train)





def confMatrix(X_train, y_train, w):

    # Add implementation here

    X_train = np.insert(X_train, 0, 1, axis=1)

    y_pred = pred(X_train, w)



    # the confusion maxtrix that we will return

    # matrix = [[0, 0], [0, 0]]

    matrix = np.zeros((2, 2), np.int8)



    # Populating our matrix using the prediction data

    for index, y in enumerate(y_train):

        if y == -1 and y_pred[index] == -1:

            matrix[0][0] += 1

        elif y == -1 and y_pred[index] == 1:

            matrix[0][1] += 1

        elif y == 1 and y_pred[index] == -1:

            matrix[1][0] += 1

        else:

            matrix[1][1] += 1



    # returning the result

    return matrix





def pred(X_train, w):

    # If X_train is not a list or array return its prediction

    if len(X_train.shape) == 1:

        # returning the dot product of w and X_train

        return np.sign(w @ X_train)



    # if X_train is an array or a list, return a prediction array corresponding to it

    else:

        # defining Y_pred as a zero vector with size of len(X_train)

        Y_pred = np.zeros(len(X_train))

        for index, x in enumerate(X_train):

            # Y_pred[i] = dot product of 'w' and 'X_train[i]'

            Y_pred[index] = np.sign(w @ x)

        # returning the result array

        return Y_pred





def test_SciKit(X_train, X_test, Y_train, Y_test):

    # instantiating the Perceptron class

    pct = Perceptron()

    # Fitting training data

    pct.fit(X_train, Y_train)

    # get the prediction of test sample

    pred_pct = pct.predict(X_test)



    # computing the confusion matrix and returning it

    return confusion_matrix(Y_test, pred_pct)





def test_Part1():

    from sklearn.datasets import load_iris

    X_train, y_train = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size=0.2)



    for i in range(80):

        if y_train[i] == 1:

            y_train[i] = -1

        else:

            y_train[i] = 1

    for j in range(20):

        if y_test[j] == 1:

            y_test[j] = -1

        else:

            y_test[j] = 1



    # Testing Part 1a

    w = fit_perceptron(X_train, y_train)

    cM = confMatrix(X_test, y_test, w)



    # Testing Part 1b

    sciKit = test_SciKit(X_train, X_test, y_train, y_test)



    print("Confusion Matrix is from Part 1a is:\n", cM)

    print("Confusion Matrix from Part 1b is:\n", sciKit)





test_Part1()



# How close is the performance of your implementation in comparison to the existing modules in the scikit-learn

# library?

# My algorithm is much more accurate than SciKit-learn package

# yeah I know, it's hard to believe eh?

# Most of the time, the results are identical,

# but sometimes, scikit-learn misclassify nearly 50% of the data

# while mine performs much more accurate. here is an example:

# Confusion Matrix is from Part 1a is:

#  [[ 9  1]

#  [ 0 10]]

# Confusion Matrix from Part 1b is:

#  [[ 0 10]

#  [ 0 10]]

# If you run this code multiple times, you will probably encounter such examples

# I owe this accuracy to the 'error_w' dictionary that I defined. I use to to memorize

# all average errors, and their correspondence 'w'. at the end I return the 'w' which yielded

# the most accurate result.