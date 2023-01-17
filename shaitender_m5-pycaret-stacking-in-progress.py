# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_cal = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
df_eval = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
df_price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
df_sample_output = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
df_sample_output.head()
df_sample_output.describe()

holiday = ['NewYear', 'OrthodoxChristmas', 'MartinLutherKingDay', 'SuperBowl', 'PresidentsDay', 'StPatricksDay', 'Easter', 'Cinco De Mayo', 'IndependenceDay', 'EidAlAdha', 'Thanksgiving', 'Christmas']
weekend = ['Saturday', 'Sunday']

df_cal['is_holiday_1'] = df_cal['event_name_1'].apply(lambda x : 1 if x in holiday else 0 )
df_cal['is_holiday_2'] = df_cal['event_name_1'].apply(lambda x : 1 if x in holiday else 0 )
df_cal['is_holiday'] = df_cal[['is_holiday_1','is_holiday_2']].max(axis=1)
df_cal['is_weekend'] = df_cal['weekday'].apply(lambda x : 1 if x in weekend else 0 )
df_cal.head()
df_cal = df_cal.drop(['weekday', 'wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], axis=1)
df_cal.head()
del_col = []
for x in range(1851):
    del_col.append('d_' + str(x+1))
df_eval = df_eval.drop(del_col, axis=1)
df_eval = df_eval.melt(['id','item_id','dept_id','cat_id','store_id','state_id'], var_name='d', value_name='qty')
df_eval = pd.merge(df_eval, df_cal, how='left', on='d')
df_eval = pd.merge(df_eval, df_price, how='left', on=['item_id', 'wm_yr_wk', 'store_id'])
df_eval_test = df_eval.query('d == "d_1852"')
df_eval_test = df_eval_test[['id', 'store_id', 'item_id', 'dept_id', 'cat_id', 'state_id', 'd', 'qty', 'sell_price']]
df_eval_test['qty'] = df_eval_test['d'].apply(lambda x: int(x.replace(x, '0')))
tmp_df = df_eval_test
for x in range(28):
    df_eval_test = df_eval_test.append(tmp_df)
    
df_eval_test = df_eval_test.reset_index(drop=True)

lst_d = []
i = 0
lst_index = df_eval_test.index
for x in lst_index:
    lst_d.append('d_' + str(((lst_index[i]) // 30490) + 1942))
    i = i + 1
    
df_eval_test['d'] = lst_d
df_eval_test = pd.merge(df_eval_test, df_cal, how='left', on='d')
df_eval_test = pd.merge(df_eval_test, df_price, how='left', on=['item_id', 'wm_yr_wk', 'store_id'])
import gc
del tmp_df
gc.collect()
df_eval = pd.get_dummies(data=df_eval, columns=['dept_id', 'cat_id', 'store_id', 'state_id'])
df_eval_test = pd.get_dummies(data=df_eval_test, columns=['dept_id', 'cat_id', 'store_id', 'state_id'])
df_eval_test = df_eval_test.drop(['sell_price_x', 'snap_CA', 'snap_TX', 'snap_WI'], axis='columns')
df_eval_test = df_eval_test.rename(columns={'sell_price_y': 'sell_price'})
df_eval = df_eval.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1)
target_col = 'qty'
exclude_cols = ['id', 'item_id', 'd', 'date', 'wm_yr_wk']
feature_cols = [col for col in df_eval.columns if col not in exclude_cols]
y = np.array(df_eval[target_col])
X = np.array(df_eval[feature_cols])
!pip install pycaret
#import regression module
from pycaret.regression import *
#intialize the setup
exp_reg = setup(df_eval[feature_cols], target = target_col)
# create individual models for stacking
lightgbm = create_model('lightgbm')
xgboost = create_model('xgboost')
ada = create_model('ada')
omp = create_model('omp')
mlp = create_model('mlp')

# stacking models
stacker = stack_models(estimator_list = [lightgbm,xgboost,ada,omp,mlp], meta_model = lightgbm)