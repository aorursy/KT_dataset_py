from multiprocessing import Pool, cpu_count

import gc; gc.enable()

import xgboost as xgb

import pandas as pd

import numpy as np

from sklearn import *

import sklearn
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/sample_submission_zero.csv')

user_logs = pd.read_csv('../input/user_logs.csv', usecols=['msno'])
user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
user_logs.head(10)