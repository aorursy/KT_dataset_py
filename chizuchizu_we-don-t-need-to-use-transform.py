import pandas as pd

import numpy as np

import dask.dataframe as dd

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn import preprocessing, metrics

import gc

import joblib

import warnings

warnings.filterwarnings('ignore')



NUM_ITEMS = 30490

DAYS_PRED = 28

nrows = 365 * 2 * NUM_ITEMS
INPUT_DIR_PATH = '../input/walmart2/'

data = pd.read_pickle(INPUT_DIR_PATH + "data22.pickle")
dedf = data.groupby(["id"])["demand"]
%%time

a = dedf.shift(7)
%%time

b = dedf.transform(lambda x: x.shift(7))
assert sum(a.fillna(0) == b.fillna(0)) == a.shape[0]  # compare between A and B
%%time

a = dedf.rolling(7).kurt().reset_index(0, drop=True).sort_index()
%%time

b = dedf.transform(lambda x: x.rolling(7).kurt())
assert sum(a.fillna(0) == b.fillna(0)) == a.shape[0]  # compare between A and B