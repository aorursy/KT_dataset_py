import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from tqdm import tqdm

import gc
# Code from https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction 



# Function to reduce the DF size

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
InputPath = "../input/ashrae-energy-prediction"

train_df = pd.read_csv(InputPath+'/train.csv')

building_df = pd.read_csv(InputPath+'/building_metadata.csv')

weather_train_df = pd.read_csv(InputPath+'/weather_train.csv')
train_df.head()
train_df.describe()
building_df.head()
building_df.describe()
weather_train_df.head()
weather_train_df.describe()
#Reduce memory usage

train_df = reduce_mem_usage(df=train_df)

weather_train_df = reduce_mem_usage(df=weather_train_df)


train = pd.merge(train_df,building_df,how = 'left')     

train = pd.merge(train,weather_train_df, on = ['site_id','timestamp'], how = 'left')

print(train.shape)



del train_df

del weather_train_df
gc.collect()
train['timestamp'] = pd.to_datetime(train.timestamp)
# Extracting date features from timestamp

train['year'] = train['timestamp'].dt.year

train['month'] = train['timestamp'].dt.month

train['day'] = train['timestamp'].dt.day

train['hour'] = train['timestamp'].dt.hour

train['dayofweek'] = train['timestamp'].dt.dayofweek
#Reduce memory usage

train = reduce_mem_usage(df=train)
train = train.drop('timestamp',axis=1)
gc.collect()
le = LabelEncoder()

train["primary_use"] = le.fit_transform(train["primary_use"])
# Convert to categorical datatype

cat_cols = ['meter', 'primary_use', 'site_id', 'building_id', 'year', 'month', 'day', 'hour', 'dayofweek']

for col in cat_cols:

    train[col] = train[col].astype('category')
target = np.log1p(train["meter_reading"])

features = train.drop('meter_reading', axis = 1)
del train
gc.collect()
features = reduce_mem_usage(df=features)
no_splits = 3

kf = KFold(no_splits)

LGBM = []

params = {

        "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 1280,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse",



}

for train,test1 in kf.split(features):

    train_features = features.loc[train]

    train_target = target.loc[train]

    

    test_features = features.loc[test1]

    test_target = target.loc[test1]

    

    training = lgb.Dataset(train_features, label=train_target,categorical_feature=cat_cols, free_raw_data=False)

    testing = lgb.Dataset(test_features, label=test_target,categorical_feature=cat_cols, free_raw_data=False)

    

    del train_features, train_target, test_features, test_target

    gc.collect()

    

    model = lgb.train(params, train_set=training, num_boost_round=1000, valid_sets=[training,testing], verbose_eval=25, early_stopping_rounds=50)

    LGBM.append(model)

    

    del training, testing

    gc.collect()



#delete intermediate dataframes

del target

del features

del train

del test1

gc.collect()
test_df = pd.read_csv(InputPath+'/test.csv')

building_df = pd.read_csv(InputPath+'/building_metadata.csv')

weather_test_df = pd.read_csv(InputPath+'/weather_test.csv')
#drop row_id in test_df



test_df = test_df.drop(columns=['row_id'])

import gc

gc.collect()
# Reduce memory usage

test_df = reduce_mem_usage(df=test_df)

weather_test_df = reduce_mem_usage(df=weather_test_df)
test = pd.merge(test_df,building_df,how = 'left')           

test = pd.merge(test,weather_test_df, on = ['site_id','timestamp'], how = 'left')

print(test.shape)
del test_df

del weather_test_df

del building_df
gc.collect()
test['timestamp'] = pd.to_datetime(test.timestamp)
test['year'] = test['timestamp'].dt.year

test['month'] = test['timestamp'].dt.month

test['day'] = test['timestamp'].dt.day

test['hour'] = test['timestamp'].dt.hour

test['dayofweek'] = test['timestamp'].dt.dayofweek
#Reduce memory usage

test = reduce_mem_usage(df=test)
#Drop timestamp from test

test = test.drop('timestamp',axis=1)
gc.collect()
le = LabelEncoder()

test["primary_use"] = le.fit_transform(test["primary_use"])
# Convert to categorical datatype

cat_cols = ['meter', 'primary_use', 'site_id', 'building_id', 'year', 'month', 'day', 'hour', 'dayofweek']

for col in cat_cols:

    test[col] = test[col].astype('category')
i=0

result=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):

    result.append(np.expm1(sum([model.predict(test.iloc[i:i+step_size]) for model in LGBM])/no_splits))

    i+=step_size

    gc.collect()
result = np.concatenate(result)
submission = pd.read_csv(InputPath+'/sample_submission.csv')

submission['meter_reading'] = result

submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0

submission.to_csv('submission.csv', index=False)