import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import warnings

warnings.filterwarnings('ignore')
## file path ##
BASE = '../input/m5-simple-fe/grid_part_1.pkl'
CALENDAR = '../input/m5-simple-fe/grid_part_3.pkl'
LAGS = '../input/m5-lags-features/lags_df_28.pkl'

SW = '../input/fast-clear-wrmsse-18ms/sw_df.pkl'
## basic features ##
df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(CALENDAR).iloc[:,9:]],
                    axis=1)

#df = pd.read_pickle(BASE)

## lag features ##
lag_df = pd.read_pickle(LAGS)
lag_df = lag_df.iloc[:, 3:11]

## input data ##
grid_df = pd.concat([df, lag_df], axis=1)

del lag_df, df
gc.collect()
## weights and scaling factors ##
# s, w, sw in sw_df are scaling factor, weight, and the product of them respectively.
# Since we use only the product sw, other columns are dropped.

sw_df = pd.read_pickle(SW)

sw_df.reset_index(inplace=True)
sw_df = sw_df[sw_df.level==11]
sw_df.drop(['level', 's', 'w'], axis=1, inplace=True)

sw_df['id'] = sw_df['id'].astype('category')
grid_df = grid_df.merge(sw_df, on='id', how='left')

# The product of sales and sw corresponds to z_it (different by a factor).
# This one is the main target.
grid_df['sw_sales'] = grid_df['sales'] * grid_df['sw']

del sw_df
gc.collect()
## training model (LGBM) ##

# train, validation and test set
START_TRAIN = 730
END_TRAIN = 1913
P_HORIZON = 28

grid_df = grid_df[grid_df.d>=START_TRAIN]
grid_df.to_pickle('grid_df_ex.pkl')

test_idx = grid_df.d > END_TRAIN
valid_idx = (grid_df.d <= END_TRAIN) & (grid_df.d > END_TRAIN - P_HORIZON)
train_idx = (grid_df.d <= END_TRAIN- P_HORIZON) & (grid_df.d >= START_TRAIN)


# hyper parameters
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'learning_rate': 0.1,
                    'num_leaves': 2**5-1,
                    'min_data_in_leaf': 2**6-1,
                    'n_estimators': 100,
                    'boost_from_average': False,
                    'verbose': -1,
                } 

# Set the metric in training Mean Absolute Error (MAE).
# Using MAE with target 'sw_sales', validation values in training show psurdo-WMASE.
lgb_params['metric'] = 'mae'
!rm train_data.bin
## Indirect Prediction ##

# features and target
remove_fe = ['id', 'd', 'sales', 'sw', 'sw_sales']
features = [fe for fe in list(grid_df) if fe not in remove_fe]

TARGET = 'sw_sales'

# dataset
train_data = lgb.Dataset(grid_df[train_idx][features], 
                        label=grid_df[train_idx][TARGET])
train_data.save_binary('train_data.bin')
train_data = lgb.Dataset('train_data.bin')

valid_data = lgb.Dataset(grid_df[valid_idx][features],
                        label=grid_df[valid_idx][TARGET])

del grid_df
gc.collect()

# model training
estimator = lgb.train(lgb_params,
                      train_data,
                      valid_sets = [train_data, valid_data],
                      verbose_eval = 10,
                      early_stopping_rounds = 5,
                      )

# Validaiton result means psuedo-WMASE at the bottom level (level 12).
# Calculated psuedo-WMASE is different by a constant factor.
# The validation score means 
## prediction for test set ##
grid_df = pd.read_pickle('grid_df_ex.pkl')

test_data = grid_df[test_idx][features]
grid_df['sw_sales'][test_idx] = estimator.predict(test_data)

del test_data
gc.collect()

# sw_sales -> sales
grid_df['sales'][test_idx] = grid_df['sw_sales'][test_idx] / grid_df['sw'][test_idx]
# The final output 'sales' accuracy is bound above at the bottom level.
# However, there are some items with weight=0.
# Some item has sales values of infinity......
grid_df['sales'][test_idx].max()