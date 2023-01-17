import numpy as np

import pandas as pd

from xgboost import XGBRegressor



import time

import sys

import gc

import pickle

sys.version_info
data = pd.read_pickle('../output/data.pkl')



data = data[[

    'date_block_num',

    'shop_id',

    'item_id',

    'item_cnt_month',

    'city_code',

    'item_category_id',

    'type_code','subtype_code',

    'item_cnt_month_lag_1','item_cnt_month_lag_2','item_cnt_month_lag_3','item_cnt_month_lag_6','item_cnt_month_lag_12',

    'item_avg_sale_last_6', 'item_std_sale_last_6',

    'item_avg_sale_last_12', 'item_std_sale_last_12',

    'shop_avg_sale_last_6', 'shop_std_sale_last_6',

    'shop_avg_sale_last_12', 'shop_std_sale_last_12',

    'category_avg_sale_last_12', 'category_std_sale_last_12',

    'city_avg_sale_last_12', 'city_std_sale_last_12',

    'type_avg_sale_last_12', 'type_std_sale_last_12',

    'subtype_avg_sale_last_12', 'subtype_std_sale_last_12',

    'date_avg_item_cnt_lag_1',

    'date_item_avg_item_cnt_lag_1','date_item_avg_item_cnt_lag_2','date_item_avg_item_cnt_lag_3','date_item_avg_item_cnt_lag_6','date_item_avg_item_cnt_lag_12',

    'date_shop_avg_item_cnt_lag_1','date_shop_avg_item_cnt_lag_2','date_shop_avg_item_cnt_lag_3','date_shop_avg_item_cnt_lag_6','date_shop_avg_item_cnt_lag_12',

    'date_cat_avg_item_cnt_lag_1',

    'date_shop_cat_avg_item_cnt_lag_1',

    'date_city_avg_item_cnt_lag_1',

    'date_item_city_avg_item_cnt_lag_1',

    'delta_price_lag',

    'month','year',

    'item_shop_last_sale','item_last_sale',

    'item_shop_first_sale','item_first_sale',

]]
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)

Y_train = data[data.date_block_num < 33]['item_cnt_month']

X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)

Y_valid = data[data.date_block_num == 33]['item_cnt_month']

X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)



del data

gc.collect();
ts = time.time()



model = XGBRegressor(

    max_depth=7,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    gamma = 0.005,

    eta=0.1,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=10, 

    early_stopping_rounds = 40,

    )



time.time() - ts
Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



X_train_level2 = pd.DataFrame({

    "ID": np.arange(Y_pred.shape[0]), 

    "item_cnt_month": Y_pred

})

X_train_level2.to_csv('../output/xgb_valid.csv', index=False)



submission = pd.DataFrame({

    "ID": np.arange(Y_test.shape[0]), 

    "item_cnt_month": Y_test

})

submission.to_csv('../output/xgb_submission.csv', index=False)