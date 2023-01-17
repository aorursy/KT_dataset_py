# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 

import datetime

import numpy as np 

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from pylab import rcParams

import warnings

from pandas.core.nanops import nanmean as pd_nanmean

from time import sleep

from tqdm.notebook import tqdm

from multiprocessing import Pool, Lock, Value



from datetime import timedelta, datetime



from sklearn.metrics import mean_absolute_error

from fbprophet import Prophet



from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression



from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



from sklearn.model_selection import TimeSeriesSplit 

warnings.filterwarnings('ignore')

%matplotlib inline



import logging

logging.disable(logging.CRITICAL)



from tqdm.contrib.concurrent import thread_map
df = pd.read_csv('/kaggle/input/sputnik/train.csv', sep =',')

df.epoch = pd.to_datetime(df.epoch, format='%Y-%m-%d %H:%M:%S')

df.index  = df.epoch

df.drop('epoch', axis = 1, inplace = True)

train, test = df[df.type == "train"], df[df.type == "test"]

train['error']  = np.linalg.norm(train[['x', 'y', 'z']].values - \

                                 train[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
train_groups = {i: train[train.sat_id == i] for i in train.sat_id.unique()}

test_groups = {i: test[test.sat_id == i] for i in test.sat_id.unique()}
def foo(test, train, k, size):

    pred = pd.DataFrame()

    for name in ('x', 'y', 'z'):

        tr = train[[name]]

        val = test[[name]]

        tr['label'] = 'train'

        val['label'] = 'val'



        df = pd.concat((tr,val), axis = 0)

        df['target'] = np.where(df.label == 'train', df[name], np.nan)



        lag_period = 24

        features = []

        for period_mult in range(1, size):

            df["lag_period_{}".format(period_mult)] = df.target.shift(period_mult*lag_period)

            features.append("lag_period_{}".format(period_mult))



        df['lagf_mean'] = df[features].mean(axis = 1)



        features.extend(['lagf_mean'])

        model = LinearRegression()

        test_df = df[df.label == 'val'][features].dropna(axis = 1)

        train_df = df[df.label == 'train'][list(test_df.columns) + ['target']].dropna()

        model.fit(train_df.drop('target', axis = 1) ,train_df['target'])

        if len(test_df) == 0:

            print(val)

        forecast = model.predict(test_df)



        test_df['prediction'] = forecast

        pred[name] = forecast

    pred.index = val.index

    return pred



doc = {}

answer = pd.DataFrame()

# for k in tqdm(range(252, 253)):

for k in tqdm(train_groups):

    train = pd.DataFrame()

    test = pd.DataFrame()

    train[['x', 'y', 'z']] = train_groups[k][['x', 'y', 'z']]

    test[['x_sim', 'y_sim', 'z_sim', 'id']] = test_groups[k][['x_sim', 'y_sim', 'z_sim', 'id']]

    size = len(train_groups[k])//24

    size = min(size, 3)

    try:

        for test_i in [test_groups[k].iloc[i*24 : (i + 1)*24] for i in range(len(test_groups[k])//24 + 1)]:

            if len(test_i) == 0:

                continue

            train = pd.concat((train, foo(test_i, train, k, size)), axis = 0)

            if size<6:

                size += 1

    except Exception:

        print(k)

        continue

    test[['x', 'y', 'z']] = train[len(train_groups[k]):]

    doc[k] = test

    doc[k]['error']  = np.linalg.norm(doc[k][['x', 'y', 'z']].values - \

                                      doc[k][['x_sim', 'y_sim', 'z_sim']].values, axis=1)

    answer = pd.concat((answer, doc[k][['id', 'error']]), axis = 0)
answer.index = np.arange(len(answer)) + 1
answer.to_csv('/kaggle/working/project.csv', index = False)