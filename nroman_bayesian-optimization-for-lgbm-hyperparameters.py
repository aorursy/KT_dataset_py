import numpy as np

import pandas as pd

import lightgbm as lgb

import warnings

import gc

from bayes_opt import BayesianOptimization

warnings.simplefilter('ignore')
building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
train = train.merge(building, on='building_id', how='left')

train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')

# Logarithmic transform of target values

y = np.log1p(train['meter_reading'])



del building, weather_train, train['meter_reading']

gc.collect();
# Transforming timestamp to a datetime format

train["timestamp"] = pd.to_datetime(train["timestamp"])



# To save time and memory this dict is going to be used to label encode primary_use feature.

# This is exactly what LabelEncoder would do to the data.

le_dict = {'Education': 0,

           'Office': 6,

           'Entertainment/public assembly': 1,

           'Lodging/residential': 4,

           'Public services': 9,

           'Healthcare': 3,

           'Other': 7,

           'Parking': 8,

           'Manufacturing/industrial': 5,

           'Food sales and service': 2,

           'Retail': 11,

           'Warehouse/storage': 15,

           'Services': 12,

           'Technology/science': 13,

           'Utility': 14,

           'Religious worship': 10}



train['primary_use'] = train['primary_use'].map(le_dict)
# Some new features from timestamp

train["month"] = train["timestamp"].dt.month

train["day"] = train["timestamp"].dt.day

train["day_of_week"] = train["timestamp"].dt.weekday

train["hour"] = train["timestamp"].dt.hour
# Saving some memory

d_types = {'building_id': np.int16,

          'meter': np.int8,

          'site_id': np.int8,

          'primary_use': np.int8,

          'square_feet': np.int32,

          'year_built': np.float16,

          'floor_count': np.float16,

          'air_temperature': np.float32,

          'cloud_coverage': np.float16,

          'dew_temperature': np.float32,

          'precip_depth_1_hr': np.float16,

          'sea_level_pressure': np.float32,

          'wind_direction': np.float16,

          'wind_speed': np.float32,

          'month': np.int8,

          'day': np.int16,

          'hour': np.int16,

          'day_of_week': np.int8}



for feature in d_types:

    train[feature] = train[feature].astype(d_types[feature])

    

gc.collect();
# building_id is useless. I am explaining it in my EDA kernel: https://www.kaggle.com/nroman/eda-for-ashrae

del train['building_id']
# By default dataset is not sorted by time. We want to train on the past data to predict future.

train = train.sort_index(by='timestamp').reset_index(drop=True)



# timestamp is no longer needed

del train['timestamp']



# Cut first 80% of the training dataset and the last 20% keep as holdout

cut_idx = int(len(train) * 0.8)

X_train, y_train, X_test, y_test = train.iloc[:cut_idx], y.iloc[:cut_idx], train.iloc[cut_idx:], y.iloc[cut_idx:]
bounds = {

    'learning_rate': (0.002, 0.2),

    'num_leaves': (50, 500), 

    'bagging_fraction' : (0.1, 1),

    'feature_fraction' : (0.1, 1),

    'min_child_weight': (0.001, 0.5),   

    'min_data_in_leaf': (20, 170),

    'max_depth': (-1, 32),

    'reg_alpha': (0.1, 2), 

    'reg_lambda': (0.1, 2)

}
def train_model(learning_rate, 

                num_leaves,

                bagging_fraction, 

                feature_fraction, 

                min_child_weight,

                min_data_in_leaf,

                max_depth,

                reg_alpha,

                reg_lambda):

    

    params = {'learning_rate': learning_rate,

              'num_leaves': int(num_leaves), 

              'bagging_fraction' : bagging_fraction,

              'feature_fraction' : feature_fraction,

              'min_child_weight': min_child_weight,   

              'min_data_in_leaf': int(min_data_in_leaf),

              'max_depth': int(max_depth),

              'reg_alpha': reg_alpha, 

              'reg_lambda': reg_lambda,

              'objective': 'regression',

              'boosting_type': 'gbdt',

              'random_state': 47,

              'verbosity': 0,

              'metric': 'rmse'}

    

    trn_data = lgb.Dataset(X_train, y_train)

    val_data = lgb.Dataset(X_test, y_test)

    model = lgb.train(params, trn_data, 5000, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds=50)

    # Returning negative rmse because optimizer tries to maximize a function

    return -model.best_score['valid_1']['rmse']
optimizer = BayesianOptimization(f=train_model, pbounds=bounds, random_state=47)

optimizer.maximize(init_points=10, n_iter=20)
print('Best RMSE score:', -optimizer.max['target'])
print('Best set of parameters:')

print(optimizer.max['params'])