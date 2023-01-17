# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from numpy import nan

from numpy import isnan

from numpy import split

from numpy import array



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import read_csv

from pandas import to_numeric



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load data

dataset = pd.read_csv('../input/household_power_consumption.txt', sep = ";", header = 0, low_memory = False, infer_datetime_format = True, parse_dates = {'datetime':[0,1]}, index_col=['datetime']) 
dataset.head()
# mark all missing values

dataset.replace('?', np.nan, inplace = True) # nan = np.nan 

# make dataset numeric

dataset = dataset.astype('float32')

dataset.head()
# fill missing value with a value at the same time oneday ago

def fill_missing(values):

    one_day = 60*24

    for row in range(values.shape[0]):

        for col in range(values.shape[1]):

            if np.isnan(values[row, col]): # isnan = np.isnan

                values[row, col] = values[row - one_day, col]
# fill_missing

fill_missing(dataset.values)
dataset.head()
# add a column for the remainder of sub_metering

values = dataset.values

dataset['sub_metering_4'] = (values[:,0] * 1000/60) - (values[:,4] + values[:,5] + values[:,6])
# save updated dataset

# dataset.to_csv('../input/household_power_consumption.csv')

# read only server

dataset.head()
daily_groups = dataset.resample('D')

daily_data = daily_groups.sum()
daily_data.head()

daily_data.shape
# evaluate one or more weekly forecasts against expected values



def evaluate_forecasts(actual, predicted):

    scores = list()

    # calculate an RMSE score for each day

    for i in range(actual.shape[0]):

        # calculate MSE

        mse = mean_squared_error(atual[:,i], predicted[:,i])

        # calculate RMSE

        rmse = sqrt(mse)

        # store

        scores.append(rmse)

    # calculate overall RMSE

    s = 0

    for row in range(actual.shape[0]):

        for col in range(actual.shape[1]):

            s += (actual[row, col] - predicted[row, col])**2

    score = sqrt(s / (actual.shape[0] * actual.shape[1]))

    return score, scores
# split a univariable dataset into train/test sets

def split_dataset(data):

    # split into standard weeks

    train, test = data[1:-328], data[-328:-6]

    # restructure into windows of weekly data

    train = array(split(train, len(train)/7))

    test = array(split(test, len(test)/7))

    return train, test
train, test = split_dataset(daily_data.values)
print(train.shape)

print(train[0,0,0], train[-1,-1,0])

print(test[0,0,0], test[-1,-1,0])
# evaluate single model

def evaluate_model(train, test, n_input):

    # fit model

    model = build_model(train, n_input)

    # history is a list of weekly data

    history = [X for X in train]

    # walk-forward validation over each week

    predictions = list()

    for i in range(len(test)):

        # predict the week

        yhat_sequence = forecast(model, history, n_input)

        # store the predictions

        predictions.append(yhat_sequence)

        # get real observation and add to history for predicting the next week

        history.append(test[i,:])

    # evaluate predictions days for each week

    predictions = array(predictions)

    score, scores = evaluate_forecasts(test[:, :, 0], predictions)

    return score, scores



# summarize scores

def summarize_scores(name, score, scores):

    s_scores = ', '.join(['%.1f' %  s for s in scores])

    print('%s: [%.3f] %s' % (name, score, s_scores))

    