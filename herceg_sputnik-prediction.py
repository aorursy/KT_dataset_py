import pandas as pd 

import datetime

import numpy as np 

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from pylab import rcParams

import warnings

from pandas.core.nanops import nanmean as pd_nanmean

from tqdm import tqdm_notebook as tqdm



from sklearn.metrics import mean_absolute_error



warnings.filterwarnings('ignore')

%matplotlib inline

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
sputnik_data = pd.read_csv("../input/sputnik/train.csv",sep=",")

sputnik_data
sputnik_train = sputnik_data.dropna(inplace=False)

sputnik_train['error']  = np.linalg.norm(sputnik_train[['x', 'y', 'z']].values - sputnik_train[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

sputnik_train.drop('x',axis=1,inplace=True)

sputnik_train.drop('y',axis=1,inplace=True)

sputnik_train.drop('z',axis=1,inplace=True)

sputnik_train.drop('x_sim',axis=1,inplace=True)

sputnik_train.drop('y_sim',axis=1,inplace=True)

sputnik_train.drop('z_sim',axis=1,inplace=True)

sputnik_train['epoch']  = pd.to_datetime(sputnik_train.epoch,format='%Y-%m-%d %H:%M:%S') 

sputnik_train.index = sputnik_train.epoch

sputnik_train.drop('epoch', axis = 1, inplace = True)
sputnik_train
def validation_forecast(sat_id):

    seasonal_periods = 24

    all_train_set = sputnik_train.loc[sputnik_train['sat_id'] == sat_id]

    all_train_set_size = all_train_set.shape[0]

    test_set = sputnik_data.loc[sputnik_data['sat_id'] == sat_id].loc[sputnik_data['type'] == 'test']

    train_size = 2 * seasonal_periods + 2

    test_size = test_set.shape[0]

    reduced_set = all_train_set.iloc[-train_size:]

    fit_first_set = ExponentialSmoothing(np.asarray(reduced_set.error),seasonal_periods = 24,trend=None, seasonal='add').fit()

    forecast = pd.Series(fit_first_set.forecast(test_size))

    forecast.index = test_set.id

    return forecast
submission = validation_forecast(0)
for i in tqdm(range(1,600)):

    submission = pd.concat([submission,validation_forecast(i)])
submission
frame = submission.to_frame()
frame.columns = ['error']
frame.to_csv('submit.csv')