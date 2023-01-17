# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from matplotlib import pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_clients = pd.read_csv('../input/x5-uplift-valid/data/clients2.csv')
df_train = pd.read_csv('../input/x5-uplift-valid/data/train.csv')
df_test = pd.read_csv('../input/x5-uplift-valid/data/test.csv')
df_products = pd.read_csv('../input/x5-uplift-valid/data/products.csv')
df_test_purch = pd.read_csv('../input/x5-uplift-valid/test_purch/test_purch.csv')
df_train_purch = pd.read_csv('../input/x5-uplift-valid/train_purch/train_purch.csv')
train_prep = df_train.copy()
train_prep['new_target'] = 0
train_prep.loc[(train_prep['treatment_flg'] == 1) & (train_prep['target'] == 1), 'new_target'] = 1
train_prep.loc[(train_prep['treatment_flg'] == 0) & (train_prep['target'] == 0), 'new_target'] = 1
train_prep
df_train_purch['transaction_datetime'] = pd.to_datetime(df_train_purch['transaction_datetime'])
train_purch_merged = pd.merge(df_train_purch, df_products, on='product_id', how='inner')
most_popular_vendor = train_purch_merged['vendor_id'].value_counts().idxmax()
train_purch_merged['most_pop_vendor'] = train_purch_merged['vendor_id'].apply(lambda x: int(x == most_popular_vendor))
most_pop_vendor_counter = train_purch_merged.groupby('client_id')['most_pop_vendor'].sum()
train_prep['vendor_count'] = train_prep['client_id'].map(most_pop_vendor_counter)
train_prep['vendor'] = train_prep['vendor_count'].apply(lambda x: int(x == 1))
netto = train_purch_merged.groupby('client_id')['netto'].sum()
train_prep['netto'] = train_prep['client_id'].map(netto)
thrsh = train_prep['netto'].median()
train_prep['netto_min'] = train_prep['netto'].apply(lambda x: int(x < 100))
day_of_week=pd.to_datetime(df_train_purch['transaction_datetime']).dt.dayofweek
train_prep['is_monday'] = (day_of_week == 0).astype('int')
most_popular_level1 = train_purch_merged['level_1'].value_counts().idxmin()
train_purch_merged['most_pop_level1'] = train_purch_merged['level_1'].apply(lambda x: int(x == most_popular_level1))
most_pop_level1 = train_purch_merged.groupby('client_id')['most_pop_level1'].sum()
train_prep['level1_count'] = train_prep['client_id'].map(most_pop_level1)
median = train_prep['level1_count'].median()
train_prep['level1_count_min'] = train_prep['level1_count'].apply(lambda x: int(x > median))
most_popular_level4 = train_purch_merged['level_4'].value_counts().idxmax()
train_purch_merged['most_pop_level4'] = train_purch_merged['level_4'].apply(lambda x: int(x == most_popular_level4))
most_pop_level4 = train_purch_merged.groupby('client_id')['most_pop_level4'].sum()
train_prep['level4_count'] = train_prep['client_id'].map(most_pop_level4)
from sklearn.model_selection import train_test_split

cols = ['vendor','netto_min',"is_monday",'level1_count_min','level4_count']
x_train, x_valid, y_train, y_valid = train_test_split(train_prep[cols], train_prep['new_target'], test_size=.2)

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score

params = {
    'n_estimators': 1000, # максимальное количество деревьев
    'max_depth': 3, # глубина одного дерева
    'learning_rate' : 0.1, # скорость обучения
    'num_leaves': 3, # количество листьев в дереве
    'min_data_in_leaf': 50, # минимальное количество наблюдений в листе
    'lambda_l1': 1, # параметр регуляризации
    'lambda_l2': 0, # параметр регуляризации
    
    'early_stopping_rounds': 10, # количество итераций без улучшения целевой метрики
}

xgb = XGBClassifier(**params)

xgb = xgb.fit(x_train, y_train, verbose=50, eval_set=[(x_valid, y_valid)], eval_metric='auc')
predicts = xgb.predict_proba(x_valid)[:, 1]

#gini
2*roc_auc_score(y_valid, predicts) - 1