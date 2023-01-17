# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import gc



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tqdm import tqdm

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold



from sklearn.externals import joblib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
calendar_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sell_prices_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

train_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sample_sub_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
calendar_df
calendar_df = calendar_df.drop(['weekday'], axis=1)

calendar_df = calendar_df.drop(['date'], axis=1)
sell_prices_df.head(10)
train_df.head(10)
sample_sub_df.head()
print(f'Shape of calendar: {calendar_df.shape}')

print(f'Shape of sell prices: {sell_prices_df.shape}')

print(f'Shape of validation dataset: {train_df.shape}')

print(f'Shape of test dataset: {sample_sub_df.shape}')
## Function to reduce the memory usage

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
calendar_df = reduce_mem_usage(calendar_df)

sell_prices_df = reduce_mem_usage(sell_prices_df)

gc.collect()
calendar_df.isna().sum()
sell_prices_df.isna().sum()
(train_df.isna().sum() == 0).all()
calendar_df = calendar_df.fillna('None')

calendar_df.isna().sum()
calendar_df.memory_usage()
calendar_df.dtypes
for feature in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:

    calendar_df[feature] = calendar_df[feature].astype('category')

    calendar_df[feature] = calendar_df[feature].cat.codes
calendar_df.memory_usage()
TOTAL_TRAINING_DAYS = 1969

TRAINING_DAYS = 1913
day_dict = {}

for i in range(TOTAL_TRAINING_DAYS):

    day_dict[f'd_{i+1}'] = i + 1

calendar_df['d'] = calendar_df['d'].map(day_dict).astype(np.int16)

calendar_df
calendar_df.dtypes
sell_prices_df.memory_usage()
for feature in ['store_id', 'item_id']:

    sell_prices_df[feature] = sell_prices_df[feature].astype('category')

    sell_prices_df[feature] = sell_prices_df[feature].cat.codes
sell_prices_df.memory_usage()
train_df.memory_usage()
train_df.dtypes
for feature in ['store_id', 'item_id', 'cat_id', 'dept_id', 'state_id']:

    train_df[feature] = train_df[feature].astype('category')

    train_df[feature] = train_df[feature].cat.codes
train_df.memory_usage()
%%time

full_train_df = train_df.drop([f'd_{i+1}' for i in range(TRAINING_DAYS)], axis=1)

full_train_df = pd.concat([full_train_df]*TRAINING_DAYS, ignore_index=True)

full_train_df['sales'] = pd.Series(train_df[[f'd_{i+1}' for i in range(TRAINING_DAYS)]].values.ravel('F'))



days = [i+1 for i in range(TRAINING_DAYS)] * len(train_df)

days.sort()



full_train_df['d'] = pd.Series(days)

full_train_df = reduce_mem_usage(full_train_df)
TEST_DAYS = 28
%%time

full_test_df = train_df.drop([f'd_{i+1}' for i in range(TRAINING_DAYS)], axis=1)

full_test_df = pd.concat([full_test_df]*TEST_DAYS, ignore_index=True)



days = [i for i in range(1914, 1914+TEST_DAYS)] * len(train_df)

days.sort()



full_test_df['d'] = pd.Series(days)

full_test_df = reduce_mem_usage(full_test_df)

full_test_df
del days

del train_df

gc.collect()
%%time

full_train_df = full_train_df.merge(calendar_df, how='inner', on='d')

full_train_df
test_cal_df = calendar_df[(calendar_df['d'] > 1913) & (calendar_df['d'] <= 1941)]

test_cal_df
%%time

full_test_df = full_test_df.merge(test_cal_df, how='inner', on='d')

full_test_df
full_train_df = reduce_mem_usage(full_train_df)

full_test_df = reduce_mem_usage(full_test_df)

gc.collect()
%%time

full_train_df = full_train_df.merge(sell_prices_df, how='inner', on=['store_id', 'item_id', 'wm_yr_wk'])

full_train_df
%%time

full_test_df = full_test_df.merge(sell_prices_df, how='inner', on=['store_id', 'item_id', 'wm_yr_wk'])

full_test_df
full_train_df = reduce_mem_usage(full_train_df)

full_test_df = reduce_mem_usage(full_test_df)

gc.collect()
del calendar_df

del sample_sub_df

del day_dict

gc.collect()
full_train_df = full_train_df.drop(['wm_yr_wk', 'd', 'id'], axis=1)

gc.collect()

full_train_df.shape
full_test_df = full_test_df.drop(['wm_yr_wk', 'd', 'id'], axis=1)

full_test_df.shape
X_train = full_train_df.drop('sales', axis=1)

Y_train = full_train_df['sales']
del full_train_df

gc.collect()
categoricals = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'wday', 'month', 'year', 'event_name_1', 

               'event_name_2', 'event_type_1', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']
params = {

      'num_leaves': 555,

      'min_child_weight': 0.034,

      'feature_fraction': 0.379,

      'bagging_fraction': 0.418,

      'min_data_in_leaf': 106,

      'objective': 'regression',

      'max_depth': -1,

      'learning_rate': 0.007,

      "boosting_type": "gbdt",

      "bagging_seed": 11,

      "metric": 'rmse',

      "verbosity": -1,

      'reg_alpha': 0.3899,

      'reg_lambda': 0.648,

      'random_state': 666,

    }

folds = 5

seed = 666



kf = StratifiedKFold(n_splits=folds, shuffle=False, random_state=seed)



models = []

for train_index, val_index in kf.split(X_train, Y_train):

    x_train = X_train.iloc[train_index]

    x_val = X_train.iloc[val_index]

    

    y_train = Y_train.iloc[train_index]

    y_val = Y_train.iloc[val_index]

    

    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)

    

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=(lgb_train, lgb_eval),

                early_stopping_rounds=100,

                verbose_eval = 100)

    

    models.append(gbm)
# save model

# joblib.dump(models, 'models.pkl')

# load model

# models = joblib.load('/kaggle/input/m5models/models.pkl')
preds = sum([model.predict(full_test_df) for model in tqdm(models)])/folds
full_test_df['sales'] = preds

full_test_df
full_test_df['item_id']
sample_sub_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

sample_sub_df
daywise_preds = {}

for day in range(TEST_DAYS):

    day_str = f'F{day+1}'

    for i in range(day, len(full_test_df), 28):

        if day_str in daywise_preds:

            daywise_preds[day_str].append(preds[i])

        else:

            daywise_preds[day_str] = [preds[i]]

            

zeros = [0 for _ in range(30490)]

for k, v in daywise_preds.items():

    daywise_preds[k] = v + zeros
daywise_preds = pd.DataFrame.from_dict(daywise_preds)
daywise_preds
daywise_preds['id'] = sample_sub_df['id']

cols = daywise_preds.columns.tolist()

cols = cols[-1:] + cols[:-1]

daywise_preds = daywise_preds[cols]
daywise_preds.to_csv('submission.csv', index=False)

daywise_preds
from IPython.display import FileLink

FileLink('submission.csv')