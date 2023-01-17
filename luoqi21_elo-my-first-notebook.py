from numpy import *

import pandas as pd

from pandas import DataFrame,Series

import matplotlib.pyplot as plt   

plt.rcParams['font.sans-serif']=['SimHei']

plt.rcParams['axes.unicode_minus']=False

import os

from pandas.io.json import json_normalize

import seaborn as sns

color = sns.color_palette()

%matplotlib inline

import lightgbm as lgb

from sklearn.model_selection import KFold

import warnings

import datetime

from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=FutureWarning)
os.listdir('../input/')
train_df = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])

test_df = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])