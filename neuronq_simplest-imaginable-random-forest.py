INPUT_DIR = '../input/'
OUTPUT_DIR = './'
!ls -lahF {INPUT_DIR}
# install my own libray of a few helpers (heavily inspired by code from the
# old version of fast.ai lib, the one used in ML1 course)
!pip install --upgrade sparklines
!pip install --upgrade treeinterpreter
!pip install --upgrade nmlu
%load_ext autoreload
%autoreload 2
%matplotlib inline

import re

import pprint
pp = pprint.PrettyPrinter(indent=2).pprint

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    cross_val_score, ShuffleSplit, train_test_split, GridSearchCV)
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor

# NMLU (Nano Machine Learning Utils) - my lib of a few simple helpers
# to keep boring code DRY
from nmlu.qinspect.nb import df_peek, df_display_all
from nmlu.qinspect.common import df_types_and_stats
import nmlu.etl as etl
import nmlu.eda as eda
import nmlu.model_analysis as ma
import nmlu.model_utils as mu

eda.set_plot_sane_defaults()
sns.set()
mpl.rcParams['figure.facecolor'] = 'white'
# load training data
df_raw = pd.read_csv(f'{INPUT_DIR}train/Train.csv', low_memory=False, 
                     parse_dates=["saledate"])
df_peek(df_raw)
# load test data (what's called "validation" on Kaggle to separate it from final test data)
df_test_raw = pd.read_csv(
    f'{INPUT_DIR}valid/Valid.csv', low_memory=False, parse_dates=["saledate"])
df_peek(df_test_raw)
# load correct results for test data ("validation solution" on Kaggle)
df_test_results = pd.read_csv(
    f'{INPUT_DIR}ValidSolution.csv')
df_peek(df_test_results)
test_y = df_test_results.SalePrice.values
test_y[:5]
def make_trn_val_data(input_df, test_sz=12000):
    df = input_df.copy()
    
    etl.train_cats(df)
    
    df.SalePrice = np.log(df.SalePrice)
        
    etl.add_datepart(df, 'saledate')
    
    x, y, nas = etl.proc_df(
        df,
        y_fld='SalePrice',
        max_n_cat=10,
        no_binary_dummies=True,
    )
    x_trn, x_val, y_trn, y_val = train_test_split(
        x, y, test_size=test_sz,
    )
    return (
        df,
        x, y, nas,
        x_trn, y_trn, x_val, y_val
    )
def make_test_data(input_df_test_raw, df_train, nas):
    df_test_raw = input_df_test_raw.copy()
    
    etl.apply_cats(df_test_raw, df_train)
    
    etl.add_datepart(df_test_raw, 'saledate')
    
    test_x, _, _ = etl.proc_df(
        df_test_raw,
        max_n_cat=10,
        no_binary_dummies=True,
        na_dict=nas.copy()
    )
    
    return df_test_raw, test_x
def rmse(x, y):
    return np.sqrt(((x - y)**2).mean())


def get_score(model, x_trn, y_trn, x_val, y_val):
    rmse_trn = rmse(m.predict(x_trn), y_trn)
    rmse_val = rmse(m.predict(x_val), y_val)
    score_trn = model.score(x_trn, y_trn)
    score_val = model.score(x_val, y_val)
    r = dict({
        "RMSE training": rmse_trn,
        "RMSE validation": rmse_val,
        "Score training": score_trn,
        "Score validation": score_val
    })
    if hasattr(model, 'oob_score_'):
        r["OOB score"] = model.oob_score_
    return r
(df, x, y, nas, x_trn, y_trn, x_val, y_val
) = make_trn_val_data(df_raw)
df_test, test_x = make_test_data(df_test_raw, df, nas)
m = RandomForestRegressor(n_estimators=100, n_jobs=-1)
m.fit(x_trn, y_trn)
get_score(m, x_trn, y_trn, x_val, y_val)
m.fit(x, y)
test_preds = m.predict(test_x)
test_result = pd.DataFrame({
    'SalesID': df_test.SalesID.values,
    'SalePrice': np.exp(test_preds)
})
# if we were to test on Kaggle (if competition were still on, we'd save this to a file)
test_result.to_csv(f'{OUTPUT_DIR}results_simple_rf.csv', index=False)
# ...but in current situation we'll just compare with the "validation solution"
test_result['TrueSalePrice'] = test_y
display(test_result.head())
print("Test RMSE:", rmse(test_preds, np.log(test_y)))
print("Test R2 Score:", m.score(test_x, np.log(test_y)))
