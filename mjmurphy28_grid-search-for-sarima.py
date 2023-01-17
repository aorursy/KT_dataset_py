# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from math import sqrt

from multiprocessing import cpu_count

from joblib import Parallel

from joblib import delayed

from warnings import catch_warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

import pandas_datareader.data as web

import warnings

warnings.simplefilter('ignore')

%matplotlib inline
def get_stock_data(ticker, start_date, end_date):

    df = web.DataReader(ticker, 'yahoo', start_date, end_date)

    df.reset_index(inplace=True)

    df = df[['Date', 'Adj Close']]

    df.columns = ['ds', 'y']

    return df
import datetime as dt

# We would like all available data from 01/01/2000 until 12/31/2016.

# can only go back to 2019-04-29 with Bloomberg

start_date = dt.datetime(2019,1,1)

end_date = dt.datetime(2019,4,29)



data = get_stock_data('TSLA', start_date, end_date)

#data = get_stock_data('AAPL', start_date, end_date)

#data = get_stock_data('AMZN', start_date, end_date)

#data = get_stock_data('GE', start_date, end_date)

#data = get_stock_data('BA', start_date, end_date)
# grid search sarima hyperparameters

from math import sqrt

from multiprocessing import cpu_count

from joblib import Parallel

from joblib import delayed

from warnings import catch_warnings

from warnings import filterwarnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error

 

# one-step sarima forecast

def sarima_forecast(history, config):

    order, sorder, trend = config

    # define model

    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)

    # fit model

    model_fit = model.fit(disp=False)

    # make one step forecast

    yhat = model_fit.predict(len(history), len(history))

    return yhat[0]



# root mean squared error or rmse

def measure_rmse(actual, predicted):

    return sqrt(mean_squared_error(actual, predicted))



# split a univariate dataset into train/test sets

def train_test_split(data, n_test):

    return data[:-n_test], data[-n_test:]



# walk-forward validation for univariate data

def walk_forward_validation(data, n_test, cfg):

    predictions = list()

    # split dataset

    train, test = train_test_split(data, n_test)

    # seed history with training dataset

    history = [x for x in train]

    # step over each time-step in the test set

    for i in range(len(test)):

        # fit model and make forecast for history

        yhat = sarima_forecast(history, cfg)

        # store forecast in list of predictions

        predictions.append(yhat)

        # add actual observation to history for the next loop

        history.append(test[i])

    # estimate prediction error

    error = measure_rmse(test, predictions)

    return error



# score a model, return None on failure

def score_model(data, n_test, cfg, debug=False):

    result = None

    # convert config to a key

    key = str(cfg)

    # show all warnings and fail on exception if debugging

    if debug:

        result = walk_forward_validation(data, n_test, cfg)

    else:

        # one failure during model validation suggests an unstable config

        try:

            # never show warnings when grid searching, too noisy

            with catch_warnings():

                filterwarnings("ignore")

                result = walk_forward_validation(data, n_test, cfg)

        except:

            error = None

    # check for an interesting result

#    if result is not None:

#        print(' > Model[%s] %.3f' % (key, result))

    return (key, result)



# grid search configs

def grid_search(data, cfg_list, n_test, parallel=True):

    scores = None

    if parallel:

        # execute configs in parallel

        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')

        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)

        scores = executor(tasks)

    else:

        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]

    # remove empty results

    scores = [r for r in scores if r[1] != None]

    # sort configs by error, asc

    scores.sort(key=lambda tup: tup[1])

    return scores



# create a set of sarima configs to try

def sarima_configs(seasonal=[0]):

    models = list()

    # define config lists

    p_params = range(2)

    d_params = range(2)

    q_params = range(5)

    t_params = ['n','c','t','ct']

    P_params = range(5)

    D_params = range(2)

    Q_params = range(2)

    m_params = range(2)

    # create config instances

    for p in p_params:

        for d in d_params:

            for q in q_params:

                for t in t_params:

                    for P in P_params:

                        for D in D_params:

                            for Q in Q_params:

                                for m in m_params:

                                    cfg = [(p, d, q), (P, D, Q, m), t]

                                    models.append(cfg)

    return models
# define dataset

#data = data['y'].values

print(data.shape)

# data split

n_test = 10

# model configs

cfg_list = sarima_configs()

print('Configurations: ', len(cfg_list))

# grid search

scores = grid_search(data['y'].values, cfg_list, n_test)

print('done')

# list top 3 configs

for cfg, error in scores[:3]:

    print(cfg, error)