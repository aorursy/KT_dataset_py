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
#import the required libraries

import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from scipy.signal import hann

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold, KFold

warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/train.csv', dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32})
#let's look at the training data

print(train_data.head())

print(100*'-')

print(train_data.shape)
#let's prepare our data

seg_length = 150000

total_samples = int(np.floor((train_data.shape[0]) / seg_length))



#we will be using a total of nine different features as given below for making our predictions

COLUMNS = ['AVERAGE', 'STD', 'MAX', 'MIN', 'SUM', 

           'AVERAGE_FIRST_10000', 'AVERAGE_LAST_10000', 'AVERAGE_FIRST_50000', 'AVERAGE_LAST_50000',

           'STD_FIRST_10000', 'STD_LAST_10000', 'STD_FIRST_50000', 'STD_LAST_50000',

           'ABS_AVERAGE', 'ABS_STD', 'ABS_MAX', 'ABS_MIN',

           '10Q', '25Q', '50Q', '75Q', '90Q', 

           'ABS_1Q', 'ABS_5Q', 'ABS_30Q', 'ABS_60Q', 'ABS_95Q', 'ABS_99Q',

           'KURTOSIS', 'SKEW', 'MEDIAN',

           'HILBERT_MEAN', 'HANN_WINDOW_MEAN']

cols = COLUMNS #our features used for the prediction

x_train = pd.DataFrame(index = range(total_samples), columns = cols, dtype = np.float64) #an empty dataframe holding our feature values

y_train = pd.DataFrame(index = range(total_samples), columns = ['time_to_failure'], dtype = np.float64) #an empty dataframe holding our target labels
for value in tqdm_notebook(range(total_samples)):

    sample = train_data.iloc[value*seg_length : value*seg_length + seg_length]

    x = pd.Series(sample['acoustic_data'].values)

    y = sample['time_to_failure'].values[-1]

    

    y_train.loc[value, 'time_to_failure'] = y

    

    x_train.loc[value, 'AVERAGE'] = x.mean()

    x_train.loc[value, 'STD'] = x.std()

    x_train.loc[value, 'MAX'] = x.max()

    x_train.loc[value, 'MIN'] = x.min() 

    x_train.loc[value, 'SUM'] = x.sum()

    

    x_train.loc[value, 'AVERAGE_FIRST_10000'] = x[:10000].mean()

    x_train.loc[value, 'AVERAGE_LAST_10000']  =  x[-10000:].mean()

    x_train.loc[value, 'AVERAGE_FIRST_50000'] = x[:50000].mean()

    x_train.loc[value, 'AVERAGE_LAST_50000'] = x[-50000:].mean()

    

    x_train.loc[value, 'STD_FIRST_10000'] = x[:10000].std()

    x_train.loc[value, 'STD_LAST_10000']  =  x[-10000:].std()

    x_train.loc[value, 'STD_FIRST_50000'] = x[:50000].std()

    x_train.loc[value, 'STD_LAST_50000'] = x[-50000:].std()

    

    x_train.loc[value, 'ABS_AVERAGE'] = np.abs(x).mean()

    x_train.loc[value, 'ABS_STD'] = np.abs(x).std()

    x_train.loc[value, 'ABS_MAX'] = np.abs(x).max()

    x_train.loc[value, 'ABS_MIN'] = np.abs(x).min()

    

    x_train.loc[value, '10Q'] = np.percentile(x, 0.10)

    x_train.loc[value, '25Q'] = np.percentile(x, 0.25)

    x_train.loc[value, '50Q'] = np.percentile(x, 0.50)

    x_train.loc[value, '75Q'] = np.percentile(x, 0.75)

    x_train.loc[value, '90Q'] = np.percentile(x, 0.90)

    

    x_train.loc[value, 'ABS_1Q'] = np.percentile(x, np.abs(0.01))

    x_train.loc[value, 'ABS_5Q'] = np.percentile(x, np.abs(0.05))

    x_train.loc[value, 'ABS_30Q'] = np.percentile(x, np.abs(0.30))

    x_train.loc[value, 'ABS_60Q'] = np.percentile(x, np.abs(0.60))

    x_train.loc[value, 'ABS_95Q'] = np.percentile(x, np.abs(0.95))

    x_train.loc[value, 'ABS_99Q'] = np.percentile(x, np.abs(0.99))

    

    x_train.loc[value, 'KURTOSIS'] = x.kurtosis()

    x_train.loc[value, 'SKEW'] = x.skew()

    x_train.loc[value, 'MEDIAN'] = x.median()

    

    x_train.loc[value, 'HILBERT_MEAN'] = np.abs(hilbert(x)).mean()

    x_train.loc[value, 'HANN_WINDOW_MEAN'] = (convolve(x, hann(150), mode = 'same') / sum(hann(150))).mean()
x_train.head() #our training dataframe holding our features
y_train.head() #our training dataframe holding our output labels
print(x_train.shape)

print(y_train.shape)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error
#normalizing the data

scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)



y_train_flatten = y_train.values.ravel() #flattening the y_train
#let's look at the mormalized data

x_train_dataframe = pd.DataFrame(x_train_scaled)

x_train_dataframe.head(10)
#import the libraries for building the neural net

import keras

from keras.layers import Dense

from keras.layers import BatchNormalization

from keras.layers import Activation

from keras.layers import Dropout

from keras.models import Sequential

from keras import optimizers
kernel_init = 'he_normal'

def Model(input_dim, activation, classes):

    model = Sequential()



#     model.add(Dense(512, kernel_initializer = kernel_init, input_dim = input_dim))

#     model.add(Activation(activation))

#     model.add(BatchNormalization())

#     model.add(Dense(32, kernel_initializer = kernel_init, input_dim = input_dim))

#     model.add(Activation(activation))

#     model.add(BatchNormalization())

#     model.add(Dropout(0.2))



    model.add(Dense(32, kernel_initializer = kernel_init)) 

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dense(32, kernel_initializer = kernel_init)) 

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))



    model.add(Dense(32, kernel_initializer = kernel_init))    

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dense(32, kernel_initializer = kernel_init))    

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(Dense(32, kernel_initializer = kernel_init))    

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dense(32, kernel_initializer = kernel_init))    

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))



    model.add(Dense(classes, kernel_initializer = kernel_init))    

    model.add(Activation('linear'))

    

    return model
input_dim = x_train.shape[1]

activation = 'tanh'

classes = 1 #the output labels

model = Model(input_dim = input_dim, activation = activation, classes = classes)



#model.summary()
#compile the model

optim = optimizers.Adam(lr = 0.001)

model.compile(loss = 'mean_absolute_error', optimizer = optim, metrics = ['mean_absolute_error'])
submission = pd.read_csv('../input/sample_submission.csv', index_col = 'seg_id')

X_test = pd.DataFrame(columns = x_train.columns, dtype = np.float64, index = submission.index)



for id in X_test.index:

    seg = pd.read_csv('../input/test/' + id + '.csv')

    

    x = pd.Series(seg['acoustic_data'].values)

    

    X_test.loc[id, 'AVERAGE'] = x.mean()

    X_test.loc[id, 'STD'] = x.std()

    X_test.loc[id, 'MAX'] = x.max()

    X_test.loc[id, 'MIN'] = x.min() 

    X_test.loc[id, 'SUM'] = x.sum()

    

    X_test.loc[id, 'AVERAGE_FIRST_10000'] = x[:10000].mean()

    X_test.loc[id, 'AVERAGE_LAST_10000']  =  x[-10000:].mean()

    X_test.loc[id, 'AVERAGE_FIRST_50000'] = x[:50000].mean()

    X_test.loc[id, 'AVERAGE_LAST_50000'] = x[-50000:].mean()

    

    X_test.loc[id, 'STD_FIRST_10000'] = x[:10000].std()

    X_test.loc[id, 'STD_LAST_10000']  =  x[-10000:].std()

    X_test.loc[id, 'STD_FIRST_50000'] = x[:50000].std()

    X_test.loc[id, 'STD_LAST_50000'] = x[-50000:].std()

    

    X_test.loc[id, 'ABS_AVERAGE'] = np.abs(x).mean()

    X_test.loc[id, 'ABS_STD'] = np.abs(x).std()

    X_test.loc[id, 'ABS_MAX'] = np.abs(x).max()

    X_test.loc[id, 'ABS_MIN'] = np.abs(x).min()

    

    X_test.loc[id, '10Q'] = np.percentile(x, 0.10)

    X_test.loc[id, '25Q'] = np.percentile(x, 0.25)

    X_test.loc[id, '50Q'] = np.percentile(x, 0.50)

    X_test.loc[id, '75Q'] = np.percentile(x, 0.75)

    X_test.loc[id, '90Q'] = np.percentile(x, 0.90)

    

    X_test.loc[id, 'ABS_1Q'] = np.percentile(x, np.abs(0.01))

    X_test.loc[id, 'ABS_5Q'] = np.percentile(x, np.abs(0.05))

    X_test.loc[id, 'ABS_30Q'] = np.percentile(x, np.abs(0.30))

    X_test.loc[id, 'ABS_60Q'] = np.percentile(x, np.abs(0.60))

    X_test.loc[id, 'ABS_95Q'] = np.percentile(x, np.abs(0.95))

    X_test.loc[id, 'ABS_99Q'] = np.percentile(x, np.abs(0.99))

    

    X_test.loc[id, 'KURTOSIS'] = x.kurtosis()

    X_test.loc[id, 'SKEW'] = x.skew()

    X_test.loc[id, 'MEDIAN'] = x.median()

    

    X_test.loc[id, 'HILBERT_MEAN'] = np.abs(hilbert(x)).mean()

    X_test.loc[id, 'HANN_WINDOW_MEAN'] = (convolve(x, hann(150), mode = 'same') / sum(hann(150))).mean()
print(X_test.shape)
#normalizing our testing data

X_test_scaled = scaler.transform(X_test)

print(X_test_scaled.shape)
input_dim = x_train.shape[1]

activation = 'tanh'

classes = 1



history = dict() #dictionery to store the history of individual models for later visualization

prediction_scores = dict() #dictionery to store the predicted scores of individual models on the test dataset



#here we will be training the same model for a total of 10 times and will be considering the mean of the output values for predictions

for i in np.arange(0, 10):

    optim = optimizers.Adam(lr = 0.001)

    ensemble_model = Model(input_dim = input_dim, activation = activation, classes = classes)

    ensemble_model.compile(loss = 'mean_absolute_error', optimizer = optim, metrics = ['mean_absolute_error'])

    print('TRAINING MODEL NO : {}'.format(i))

    H = ensemble_model.fit(x_train_scaled, y_train_flatten,

                  batch_size = 64,

                  epochs = 200,

                  verbose = 1)

    history[i] = H

    

    ensemble_model.save('MODEL_{}.model'.format(i))

    

    predictions = ensemble_model.predict(X_test_scaled, verbose = 1, batch_size = 64)

    prediction_scores[i] = predictions
#making predictions

prediction1 = np.hstack([p.reshape(-1,1) for p in prediction_scores.values()]) #taking the scores of all the trained models

prediction1 = np.mean(prediction1, axis = 1)



print(prediction1.shape)
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
# fixed parameters

params = {'boosting_type': 'gbdt',

          'objective': 'regression_l1',

          'nthread': 4,

          'num_leaves': 64,

          'learning_rate': 0.05,

          'max_bin': 512,

          'colsample_bytree': 1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.1,

          'min_split_gain': 0.1,

          'min_child_weight': 0.1,

          'min_child_samples': 20,

          'metric' : 'mae'}
gbm = lgb.LGBMRegressor(boosting_type= 'gbdt',

                        objective = 'regression_l1',

                        n_jobs = 4,

                        silent = True,

                        max_bin = params['max_bin'],

                        min_split_gain = params['min_split_gain'],

                        min_child_weight = params['min_child_weight'],

                        min_child_samples = params['min_child_samples'])
folds = KFold(n_splits = 12, shuffle = True, random_state = 101)



for fold_n, (train_index, valid_index) in enumerate(folds.split(x_train)):

    print('Fold', fold_n)

    X_train_f, X_valid = x_train.iloc[train_index], x_train.iloc[valid_index]

    y_train_f, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]



    gbm.fit(X_train_f, y_train_f.values.flatten(),

            eval_set=[(X_valid, y_valid.values.flatten())],

            eval_metric = 'mae',

            early_stopping_rounds = 200)



    print('Starting predicting...')

    # predict

    y_pred = gbm.predict(X_valid, num_iteration = gbm.best_iteration_)

    # eval

    print('The mae of prediction is:', mean_absolute_error(y_valid.values.flatten(), y_pred))



# feature importances

print('Feature importances:', list(gbm.feature_importances_))
#sunmitting the file

submission['time_to_failure'] = prediction1

submission.to_csv('ensemble_submission.csv')
lgbm_predictions = gbm.predict(X_test)
submission['time_to_failure'] = lgbm_predictions

print(submission.head())

submission.to_csv('lgbm_submission.csv')
submission['time_to_failure'] = 0.5 * lgbm_predictions + 0.5 * prediction1

submission.to_csv('lgbm_ensemble_submission.csv')
submission['time_to_failure'] = 0.2 * lgbm_predictions + 0.8 * prediction1

submission.to_csv('0.2lgbm_ensemble_0.8submission.csv')
submission['time_to_failure'] = 0.8 * lgbm_predictions + 0.2 * prediction1

submission.to_csv('0.8lgbm_ensemble_0.2submission.csv')