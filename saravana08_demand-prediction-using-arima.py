import pandas as pd   # For Dataframe

import matplotlib.pyplot as plt  # for creating Plots

%matplotlib inline 

import warnings

import math

from pandas import read_csv

from pandas import datetime

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

import warnings

import itertools

import pandas as pd

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from pandas import DataFrame

from matplotlib import pyplot



train = pd.read_csv("../input/Data.csv",index_col=[0])



# evaluate an ARIMA model for a given order (p,d,q)

def evaluate_arima_model(X, arima_order):

    # prepare training dataset

    train_size = int(len(X) * 0.75)

    train, test = X[0:train_size], X[train_size:]

    history = [x for x in train]

    # make predictions

    predictions = list()

    for t in range(len(test)):

        model = ARIMA(history, order=arima_order)

        model_fit = model.fit(disp=False,transparams=True,trend='c',)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(test[t])

    # calculate out of sample error

    error = math.sqrt(mean_squared_error(test, predictions))

    return error





# evaluate combinations of p, d and q values for an ARIMA model

def evaluate_models(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float32')

    global best_cfg

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    rmse = evaluate_arima_model(dataset, order)

                    if rmse < best_score:

                        best_score, best_cfg = rmse, order

                    print('ARIMA%s RMSE=%.3f' % (order,rmse))

                except:

                    continue

    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

    

    

p_values = range(0, 8)

d_values = range(0, 4)

q_values = range(0, 4)

warnings.filterwarnings("ignore")

evaluate_models(train.values, p_values, d_values, q_values)

    

train = train.astype('float32')

mod1 = ARIMA(train,order=(best_cfg))

results = mod1.fit()

print(results.summary())



train.plot(figsize=(8, 6))

plt.show()

train = train.astype('float32')

train_size = int(len(train) * 0.75)

train, test = train[0:train_size], train[train_size:]





residuals = DataFrame(results.resid)

residuals.plot()

pyplot.show()

residuals.plot(kind='kde')

pyplot.show()

print(residuals.describe())
