import numpy as np

import pandas as pd

from numpy import isnan

from math import sqrt



train = pd.read_csv("../input/df_train.csv")

test= pd.read_csv("../input/df_test.csv")

df_train = pd.DataFrame(train)

df_test = pd.DataFrame(test)



def transformNanValues(feature_variable):

    where_are_NaNs = isnan(feature_variable)

    df_train[where_are_NaNs] = 0

    

transformNanValues(df_train["x"])

transformNanValues(df_train["y"])

# Calculate the mean value of a list of numbers

def mean(values):

    return sum(values) / float(len(values))

# Calculate covariance between x and y

def covariance(x, mean_x, y, mean_y):

    covar = 0.0

    for i in range(len(x)):

        covar += (x[i] - mean_x) * (y[i] - mean_y)

    return covar

# Calculate the variance of a list of numbers

def variance(values, mean):

    return sum([(x-mean)**2 for x in values])

def coefficients(df):

    x = df["x"].tolist()

    y = df["y"].tolist()

    x_mean, y_mean = mean(x), mean(y)

    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)

    b0 = y_mean - b1 * x_mean

    return [b0, b1]

# Calculate root mean squared error

def rmse_metric(actual, predicted):

    sum_error = 0.0

    for i in range(len(actual)):

        prediction_error = predicted[i] - actual[i]

        sum_error += (prediction_error ** 2)

        mean_error = sum_error / float(len(actual))

    return sqrt(mean_error)

# Evaluate regression algorithm on training dataset

def evaluate_algorithm(algorithm,  *args):

    train = df_train;

    test = df_test;

    test_set = list()

    for row in test:

        row_copy = list(row)

        row_copy[-1] = None

        test_set.append(row_copy)

    predicted = algorithm(train, test_set, *args)

    actual = test["y"]

    # print("actual values are", actual)

    rmse = rmse_metric(actual, predicted)

    return rmse

def simple_linear_regression(train, test):

    predictions = list()

    b0, b1 = coefficients(train)

    for row in df_test["x"]:

        yhat = b0 + b1 * row

        predictions.append(yhat)

    # print("predictions are =>", predictions)

    return predictions

rmse = evaluate_algorithm(simple_linear_regression)

print('RMSE: %.3f' % (rmse))


