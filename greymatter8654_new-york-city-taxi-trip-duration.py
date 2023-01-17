%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import timedelta

import datetime as dt

from haversine import haversine

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans
np.random.seed(1987)

N = 100000 # number of sample rows in plots

t0 = dt.datetime.now()

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
print('We have {} training rows and {} test rows.'.format(train.shape[0], test.shape[0]))

print('We have {} training columns and {} test columns.'.format(train.shape[1], test.shape[1]))

train.head(2)
print('Id is unique.') if train.id.nunique() == train.shape[0] else print('oops')

print('Train and test sets are distinct.') if len(np.intersect1d(train.id.values, test.id.values))== 0 else print('oops')

print('We do not need to worry about missing values.') if train.count().min() == train.shape[0] and test.count().min() == test.shape[0] else print('oops')

print('The store_and_fwd_flag has only two values {}.'.format(str(set(train.store_and_fwd_flag.unique()) | set(test.store_and_fwd_flag.unique()))))