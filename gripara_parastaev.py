import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# some auxiliary libraries

import datetime

from tqdm import tqdm

from pylab import rcParams

import warnings



warnings.filterwarnings('ignore')

%matplotlib inline



# method of scikit-learn package for time series validation

from sklearn.model_selection import TimeSeriesSplit
# function for SMAPE calculation

def smape_score(y_true, y_pred):

    return np.mean(2 * np.abs(y_true - y_pred) / 

                   (np.abs(y_true) + np.abs(y_pred))) * 100
sat_pos = pd.read_csv('/kaggle/input/sputnik/train.csv', sep=',')

sat_pos['epoch'] = pd.to_datetime(sat_pos.epoch, format='%Y-%m-%d %H:%M:%S')

sat_pos.index = sat_pos.epoch

sat_pos.drop('epoch', axis=1, inplace=True)



sat_pos['error'] = np.linalg.norm(sat_pos[['x', 'y', 'z']].values - 

                                        sat_pos[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

sat_pos.drop(['x', 'y', 'z', 'x_sim', 'y_sim', 'z_sim'], axis=1, inplace=True)



np.array_equal(np.unique(sat_pos['sat_id']), np.arange(0,600))



sat_set = np.arange(0,600)

sat_set.shape[0]

sat_pos_train = sat_pos[sat_pos['type'] == 'train']

sat_pos_test = sat_pos[sat_pos['type'] == 'test']



sat_pos_train.drop('type', axis=1, inplace=True)
def timeseriesCVscore(series, n_splits, loss_function):

    

    errors = []

    tscv = TimeSeriesSplit(n_splits=n_splits)



    for train, test in tscv.split(series.error.values):

 

        x_true = series.error.values[train]

        y_true = series.error.values[test]



        l_train = len(x_true)

        l_test = len(y_true)

        lags_errors = []

        lag_period = 24

        n_steps = l_test // lag_period

        remainder = l_test % lag_period

        for step in np.arange(1, n_steps + 1):

            lag_error = (series.error.shift(lag_period) + 

                        step * (series.error.shift(lag_period) - 

                                  series.error.shift(2*lag_period)))[l_train:l_train+lag_period]

            lags_errors.append(lag_error)

        if remainder != 0:

            lag_error = (series.error.shift(lag_period) + 

                        (n_steps + 1) * (series.error.shift(lag_period) - 

                                      series.error.shift(2*lag_period)))[l_train:l_train+remainder]

            lags_errors.append(lag_error)   

        

        y_pred = np.concatenate(lags_errors)

        error = loss_function(y_pred, y_true)

        errors.append(error)

    

    return np.mean(np.array(errors))
series = sat_pos_train[sat_pos_train['sat_id'] == 150]

n_splits = 100

timeseriesCVscore(series=series, n_splits=n_splits, loss_function=smape_score)
train_lengths = []

test_lengths = []



for sat_i in tqdm(sat_set):

    series = sat_pos[sat_pos['sat_id'] == sat_i]

    l_train = len(series[series['type'] == 'train'])

    l_test = len(series[series['type'] == 'test'])

    train_lengths.append(l_train)

    test_lengths.append(l_test)



len_train_max = max(train_lengths)

# Check that all trainsets are greater in size than testsets

#res_gt = [*map(lambda x: x[0]>x[1], zip(train_lengths, test_lengths))]

#res_gt
predictions = []



for sat_i in tqdm(sat_set):

    series = sat_pos[sat_pos['sat_id'] == sat_i]

    l_train = len(series[series['type'] == 'train'])

    l_test = len(series[series['type'] == 'test'])

    lags_errors = []

    lag_period = 24

    n_steps = l_test // lag_period

    remainder = l_test % lag_period

    for step in np.arange(1, n_steps + 1):

        lag_error = (series.error.shift(lag_period) + 

                     step * (series.error.shift(lag_period) -

                             series.error.shift(2*lag_period)))[l_train:l_train+lag_period]

        lags_errors.append(lag_error)

    if remainder != 0:

        lag_error = (series.error.shift(lag_period) + 

                     (n_steps + 1) * (series.error.shift(lag_period) -

                                      series.error.shift(2*lag_period)))[l_train:l_train+remainder]

        lags_errors.append(lag_error)

    predictions.append(np.concatenate(lags_errors))
forecast = np.concatenate(predictions)



subm = pd.DataFrame({'id': sat_pos_test['id'],

             'error': forecast})



subm.to_csv('submission_3_5.csv', index=False)