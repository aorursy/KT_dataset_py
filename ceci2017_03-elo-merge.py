# This Python 3 environment comes with many helpful analytics libraries installedimport numpy as np

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

import seaborn as sns

from tqdm import tqdm

from datetime import datetime

import json

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split, KFold, cross_val_score 

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import LabelEncoder, Imputer

from sklearn import model_selection, preprocessing, metrics

import lightgbm as lgb



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from scipy.stats import skew 

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

# models

from xgboost import XGBRegressor

import warnings



# Ignore useless warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")



# Avoid runtime error messages

pd.set_option('display.float_format', lambda x:'%f'%x)



# make notebook's output stable across runs

np.random.seed(42)



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



import os

print(os.listdir("../input"))
test_df = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
hist_df = pd.read_csv("../input/historical_transactions.csv")
hist_df['category_2'].fillna(hist_df['category_2'].mode()[0], inplace=True)
hist_df['category_3'].fillna(hist_df['category_3'][0], inplace=True)
hist_df['merchant_id'].fillna(hist_df['merchant_id'][0], inplace=True)
hist_df = hist_df[hist_df['authorized_flag'] == 'Y']
new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv")

new_trans_df.head()
new_trans_df['category_3'].fillna(new_trans_df['category_3'][0], inplace=True)
new_trans_df['category_2'].fillna(new_trans_df['category_2'].mode()[0], inplace=True)
new_trans_df['merchant_id'].fillna(new_trans_df['merchant_id'][0], inplace=True)
all_new_hist = pd.concat([new_trans_df,hist_df], axis=0, ignore_index=True)
all_new_hist.to_csv("all_new_hist.csv", index=False)
#all_df2 = pd.merge(test_df, all_new_hist, on='card_id', how='inner')
#listGroup = all_df2.groupby(['card_id'])['card_id'].count()
#listGroup.shape

#123623 registros
#all_df2.to_csv("all_df2.csv", index=False)