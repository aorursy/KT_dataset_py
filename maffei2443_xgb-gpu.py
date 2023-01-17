%%time
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns
import itertools
import gc
import pickle
import os
from time import time

import pathlib
print("Pasta atual:", pathlib.Path().absolute())

# from catboost import CatBoostRegressor, Pool, cv
# from catboost.utils import get_gpu_device_count

# nome do modelo e do .csv
from datetime import datetime as dtime
tname = str(dtime.now())[:-7]


# %%time
path='../input/kddbr-2020/'
def load_year(y):
    dtypes = pickle.load(open(
        f'../input/kdd2020-cpr/{y}_dtypes.pkl', 'rb'
    ))
    del dtypes['date']
    df = pd.read_csv(
        f'../input/kdd2020-cpr/{y}.csv',
        dtype=dtypes, parse_dates=['date'], index_col='id'
    )
    return df

YEARS = [2018]
%%time
base = [load_year(year) for year in YEARS]
base = pd.concat(base)

columns_size = None
rows_size = None
if (columns_size != None or rows_size != None):
    inputs = list( base.columns[ base.columns.str.contains('input')][:columns_size])
    inputs.sort()
    output = list(base.columns[ base.columns.str.contains('output')] )
    cols = list( base[ inputs + output  ].columns )
    cols.append('id')
    cols.append('date')
    base = base[cols].copy()[:rows_size]

# %%time
input_columns = base.columns[base.columns.str.contains('input') ]
output_columns = base.columns[base.columns.str.contains('output') ]

print(F'Inputs: {input_columns.shape} Outputs: {output_columns.shape}')
# input_columns, output_columns


# %%time
def create_features(df):
    df['input_month'] = df.date.dt.month
    df['input_year'] = df.date.dt.year
    df['input_day'] = df.date.dt.day
    df['input_dt_sin_quarter']     = np.sin(2*np.pi*df.date.dt.quarter/4)
    df['input_dt_sin_day_of_week'] = np.sin(2*np.pi*df.date.dt.dayofweek/6)
    df['input_dt_sin_day_of_year'] = np.sin(2*np.pi*df.date.dt.dayofyear/365)
    df['input_dt_sin_day']         = np.sin(2*np.pi*df.date.dt.day/30)
    df['input_dt_sin_month']       = np.sin(2*np.pi*df.date.dt.month/12)
    return df

def is_weekend(num):
    return num > 5

%%time
def date_expand(df, pipeline = True):
    df['date'] = pd.to_datetime( df.date )
    dt = df['date'].dt
    df['week'] = dt.week
    df['weekday'] = dt.weekday + 1
    df['weekday_sin'] = np.sin(2*np.pi*df.date.dt.weekday/7)
    
    df['weekofyear'] = dt.weekofyear
    df['weekofyear_sin'] = np.sin(2*np.pi*df.date.dt.weekofyear/52)

    df['weekend'] = dt.weekday.apply(is_weekend)
    return df if pipeline else None

create_features(base);
date_expand(base);
input_columns = base.columns[base.columns.str.contains('input') ]
output_columns = base.columns[base.columns.str.contains('output') ]
%%time
# necessário pois o MultiOutputRegresso espera que não haja NaN nos dados
# Obviamente, 6358 NÃO está presente em 2018
CUSTOM_NA = 6358


# %%time
X = base[input_columns[:columns_size] ].fillna(CUSTOM_NA).values
Y = base[output_columns].fillna(0).values

from sklearn.model_selection import train_test_split
# x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.0, random_state=42)


from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

model = XGBRegressor(
    n_estimators=7, random_state=0,
    learning_rate=.1, max_depth=1, colsample_bytree=.8, colsample_bynode=.9,
    tree_method='gpu_hist', sampling_method='gradient_based',
    gpu_id=0, verbosity=1, missing=CUSTOM_NA
)

clf = MultiOutputRegressor(model)
clf.fit(X[:3],Y[:3])
%%time
QTD = None
#Loading test file
# _df = load_year(2019)

# _df = create_features(_df)
input_columns = _df.columns[_df.columns.str.contains('input') ]
_df = _df[input_columns].copy()
_df.fillna(CUSTOM_NA, inplace=True)

# %%time
pred = clf.predict(_df.values[:QTD])
pred_sub = pd.DataFrame(pred)
pred_sub.columns = output_columns
pred_sub['id'] = _df.index[:QTD]


# %%time
submission = []
for i, row in pred_sub.iterrows():
    for column, value in zip(output_columns, row.values):
        _id = "{}_{}".format(int(row.id), column)
        submission.append([_id, value])
    break

del _df, test
gc.collect()
submission = pd.DataFrame(submission)
submission.columns = ['id', 'value']
submission.to_csv('{}.csv'.format(tname), index=False)

try:
    os.makedirs('clfs/xgb', exist_ok=True)
    pickle.dump(clf, open('clfs/xgb/{}.pkl'.format(tname), 'wb') )
except:
    print('erro no pickle')
