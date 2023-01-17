INPUT_DIR = '../input/'
OUTPUT_DIR = './'
!ls -lah {INPUT_DIR}
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
import matplotlib.pyplot as plt
import seaborn as sns  # do sns.set() if you want its default styles globally applied

from sklearn.model_selection import (
    cross_val_score, ShuffleSplit, train_test_split, GridSearchCV)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor

# NMLU (Nano Machine Learning Utils) - my lib of a few simple helpers to keep boring code DRY
from nmlu.qinspect.nb import df_peek, df_display_all
from nmlu.qinspect.common import df_types_and_stats
import nmlu.etl as etl
import nmlu.eda as eda
import nmlu.model_analysis as ma
import nmlu.model_utils as mu

eda.set_plot_sane_defaults()
df_raw = pd.read_csv(f'{INPUT_DIR}train.csv', low_memory=False)
df_peek(df_raw, 'df_raw')  # summary with stats and datatypes
# confirm ids unique
df_raw.Id.nunique(), df_raw.shape
# price to log
df_raw.SalePrice = np.log(df_raw.SalePrice)
# categories
etl.train_cats(df_raw)
df_peek(df_raw)
# make fully numericalize version of data
nzDf_raw = etl.numericalized_df(df_raw)
pp(nzDf_raw.dtypes)
nzDf_raw[:5]
print(list(df_raw.columns))
sns.pairplot(
        nzDf_raw,
        vars=['SalePrice', 'YrSold', 'MoSold', 'LotArea', 'BedroomAbvGr'],
    )
def rmse(x,y): return np.sqrt(((x-y)**2).mean())

def get_score(model, x_trn, y_trn, x_val, y_val):
    rmse_trn = rmse(model.predict(x_trn), y_trn)
    rmse_val = rmse(model.predict(x_val), y_val)
    r = dict({
        "RMSE training": rmse_trn,
        "RMSE validation": rmse_val,
        "R2 training": model.score(x_trn, y_trn),
        "R2 validation": model.score(x_val, y_val)
    })
    if hasattr(model, 'oob_score_'):
        r["OOB score"] = model.oob_score_
    return r
# simplest non-useless model we can think of,
# to get some feature importances
x, y, nas = etl.proc_df(
    df_raw, 'SalePrice', skip_flds=['Id'])

x_trn, x_val, y_trn, y_val = train_test_split(
    x, y, test_size=int(0.8 * len(x)))

model = RandomForestRegressor(
    n_estimators=100,
#     max_depth=14,  # ~ log2(features) * 2
    oob_score=True,
)

model.fit(x_trn, y_trn)
pp(get_score(model, x_trn, y_trn, x_val, y_val))
# repeat model training and feature importance to make sure
# we don't have a model so random it's useless
for i in range(4):
    model.fit(x_trn, y_trn)
    pp(get_score(model, x_trn, y_trn, x_val, y_val))
    ma.rf_show_plot_fi(model, x_val, top_n=20)
# also test predictions of this simple model
df_test = pd.read_csv(f'{INPUT_DIR}test.csv')

etl.apply_cats(df_test, df_raw)

# df_test.SalePrice = np.log(df_test.SalePrice)

# df_peek(df_test)

x_test, y_test, nas_test = etl.proc_df(df_test, na_dict=nas.copy())

model.fit(x, y)

print("OOB score:", model.oob_score_)

preds_final = model.predict(x_test.drop(columns='Id'))
result = pd.DataFrame({'Id': x_test.Id,
                       'SalePrice': np.exp(preds_final)})
display(result.head())

result.to_csv(f'{OUTPUT_DIR}results_eda.csv', index=False)
preds_final
ohe_x, ohe_y, ohe_nas = etl.proc_df(
    df_raw, 'SalePrice',
    max_n_cat=6,
    no_binary_dummies=True,
    skip_flds=['Id']
)

ohe_x_trn, ohe_x_val, ohe_y_trn, ohe_y_val = train_test_split(
    ohe_x, ohe_y, test_size=int(0.8 * len(x)))

ohe_model = RandomForestRegressor(
    n_estimators=100,
#     max_depth=14,  # ~ log2(features) * 2
    oob_score=True,
)

ohe_model.fit(ohe_x_trn, ohe_y_trn)
pp(get_score(ohe_model, ohe_x_trn, ohe_y_trn, ohe_x_val, ohe_y_val))
# repeat model training and feature importance to make sure
# we don't have a model so random it's useless (OHE version)
for i in range(4):
    ohe_model.fit(ohe_x_trn, ohe_y_trn)
    pp(get_score(ohe_model, ohe_x_trn, ohe_y_trn, ohe_x_val, ohe_y_val))
    ma.rf_show_plot_fi(ohe_model, ohe_x_val, top_n=20)
# also test predictions of this simple model
ohe_df_test = pd.read_csv(f'{INPUT_DIR}test.csv')

etl.apply_cats(ohe_df_test, df_raw)

# df_test.SalePrice = np.log(df_test.SalePrice)

# df_peek(df_test)

ohe_x_test, ohe_y_test, ohe_nas_test = etl.proc_df(
    ohe_df_test, na_dict=ohe_nas.copy())

ohe_model.fit(ohe_x, ohe_y)

print("OOB score:", ohe_model.oob_score_)

ohe_preds_final = model.predict(ohe_x_test.drop(columns='Id'))
ohe_result = pd.DataFrame({'Id': ohe_x_test.Id,
                           'SalePrice': np.exp(ohe_preds_final)})
display(ohe_result.head())

result.to_csv(f'{OUTPUT_DIR}results_eda_ohe.csv', index=False)