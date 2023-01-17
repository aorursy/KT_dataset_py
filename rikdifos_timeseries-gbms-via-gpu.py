%matplotlib inline

%config InlineBackend.figure_format = 'svg' 

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)



import matplotlib.pyplot as plt



from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from xgboost import plot_importance



import time, sys, gc, pickle



#from itertools import product

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import TimeSeriesSplit
start = time.time()

data = pd.read_pickle('../input/feature-engineering-xgboost/data.pkl')

print('data input costs {:.2f} seconds'.format(time.time()-start))

print('data has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')

print('test has {} rows and {} columns'.format(test.shape[0], test.shape[1]))
data = data[[

    'date_block_num',

    'shop_id',

    'item_id',

    'item_cnt_month',

    'city_code',

    'item_category_id',

    'type_code',

    'subtype_code',

    'item_cnt_month_lag_1',

    'item_cnt_month_lag_2',

    'item_cnt_month_lag_3',

    'item_cnt_month_lag_6',

    'item_cnt_month_lag_12',

    'date_avg_item_cnt_lag_1',

    'date_item_avg_item_cnt_lag_1',

    'date_item_avg_item_cnt_lag_2',

    'date_item_avg_item_cnt_lag_3',

    'date_item_avg_item_cnt_lag_6',

    'date_item_avg_item_cnt_lag_12',

    'date_shop_avg_item_cnt_lag_1',

    'date_shop_avg_item_cnt_lag_2',

    'date_shop_avg_item_cnt_lag_3',

    'date_shop_avg_item_cnt_lag_6',

    'date_shop_avg_item_cnt_lag_12',

    'date_cat_avg_item_cnt_lag_1',

    'date_shop_cat_avg_item_cnt_lag_1',

    #'date_shop_type_avg_item_cnt_lag_1',

    #'date_shop_subtype_avg_item_cnt_lag_1',

    'date_city_avg_item_cnt_lag_1',

    'date_item_city_avg_item_cnt_lag_1',

    #'date_type_avg_item_cnt_lag_1',

    #'date_subtype_avg_item_cnt_lag_1',

    'delta_price_lag',

    'month',

    'days',

    'item_shop_last_sale',

    'item_last_sale',

    'item_shop_first_sale',

    'item_first_sale',

]]



#data = data.sort_values('date_block_num')

x_train = data[data.date_block_num <= 33].drop(['item_cnt_month'], axis=1)

y_train = data[data.date_block_num <= 33]['item_cnt_month']

#X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)

#Y_valid = data[data.date_block_num == 33]['item_cnt_month']

x_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)



del data

gc.collect();



def run_cross_validate(x_train, y_train, test, N_FOLDS=5):

    model = XGBRegressor(

        objective='reg:squarederror',

        learning_rate=0.05, 

        max_depth=8,

        n_estimators=1000,

        min_child_weight=200, 

        colsample_bytree=0.8, 

        subsample=0.8, 

        seed=42,

        tree_method='gpu_hist' # turn GPU on!!!

    )

    tspl = TimeSeriesSplit(n_splits=N_FOLDS)

    oof_preds = np.zeros(len(x_train))                      # 交叉验证预测结果

    test_preds = np.zeros(len(test))                        # 测试集预测结果

    oof_losses = []

    print('gpu training begins ...')

    for fold, (trn_idx, val_idx) in enumerate(tspl.split(x_train, y_train)):

        print('Starting fold: ', fold + 1)

        x_trn, x_val = x_train.iloc[trn_idx], x_train.iloc[val_idx]

        y_trn, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]



        model.fit(x_trn, y_trn, 

                    eval_metric='rmse', 

                    eval_set=[(x_trn, y_trn),(x_val, y_val)], 

                    verbose=10, 

                    early_stopping_rounds = 10)             # set early stopping

        val_preds = model.predict(x_val) 

        oof_preds[val_idx] = val_preds

        loss = np.sqrt(mean_squared_error(y_val, val_preds))

        print('fold {} RMSE is {:.5f}'.format(fold + 1, loss))

        oof_losses.append(loss)

        preds = model.predict(test) 

        test_preds += preds / N_FOLDS

        print('-' * 50)

        print('\n')

    print('Mean OOF RMSE across folds is {:.5f}'.format(np.mean(oof_losses))) # 每一折的验证分数平均

    print('GPU Xgb costs {:.2f} seconds'.format(time.time()-start))

    return test_preds, oof_preds
test_preds, oof_preds = run_cross_validate(x_train, y_train, x_test, 3)
Y_test = test_preds.clip(0, 20)

submission = pd.DataFrame({ 'ID': test.index, 'item_cnt_month': Y_test})

submission.to_csv('submission.csv', index=False)
'''

start = time.time()

print('cpu training begins ...')

model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42

)



model.fit(

    X_train, 

    Y_train, 

    eval_metric='rmse', 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)



print('CPU Xgb costs {:.2f} seconds'.format(time.time()-start))

'''
'''

start = time.time()

print('gpu training begins ...')

model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42,

    tree_method='gpu_hist' # turn GPU on!!!

)



model.fit(

    X_train, 

    Y_train, 

    eval_metric='rmse', 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=10, 

    early_stopping_rounds = 10)



print('GPU Xgb costs {:.2f} seconds'.format(time.time()-start))

'''
'''

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



plt.rcParams['figure.facecolor'] = 'white'

plot_features(model, (7.5,10))

'''
'''

print('learning rate:',model.learning_rate_)

Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({ 'ID': test.index, 'item_cnt_month': Y_test})

submission.to_csv('submission.csv', index=False)

'''
'''

start = time.time()

print('gpu training begins ...')

model = CatBoostRegressor(eval_metric='RMSE',

                          iterations=1000,

                          max_ctr_complexity=4,

                          random_seed=42,

                          od_type='Iter',

                          od_wait=100,

                          verbose=50,

                          depth=8,

                          metric_period = 50,

                          task_type='GPU'

)

Y_train = np.array(Y_train).astype('float32')

Y_valid = np.array(Y_valid).astype('float32')

model.fit(X_train, 

          Y_train,

          eval_set=[(X_valid, Y_valid)], 

          verbose=True, 

          use_best_model=True)

print('GPU Cat costs {:.2f} seconds'.format(time.time()-start))    

'''
# save predictions for an ensemble

# pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))

# pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))