import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import datetime as dt 

%matplotlib inline

import matplotlib.pyplot as plt

from tqdm import tqdm

import lightgbm as lgb

import datetime as dt



pd.options.display.float_format = '{:.4f}'.format
root = "../input/ashrae-energy-prediction/"



#read data from csv files

train = pd.read_csv(root + 'train.csv')

test = pd.read_csv(root + 'test.csv')

w_train = pd.read_csv(root + 'weather_train.csv', index_col=False)

w_test = pd.read_csv(root + 'weather_test.csv', index_col=False)

building = pd.read_csv(root + 'building_metadata.csv')



#change the categorical primary use column to numerical with label encoder

le = LabelEncoder()

building.primary_use = le.fit_transform(building.primary_use)

## Memory Reducer

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

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
reduce_mem_usage(train, verbose=True)

reduce_mem_usage(test, verbose=True)

reduce_mem_usage(w_train, verbose=True)

reduce_mem_usage(building, verbose=True)

reduce_mem_usage(w_test, verbose=True)
train = train.merge(building, on='building_id', how='left')

train = train.merge(w_train, on=['site_id', 'timestamp'], how='left')
test = test.merge(building, on='building_id', how='left')

test = test.merge(w_test, on=['site_id', 'timestamp'], how='left')
# We will need this row for our submission, 

#but it is not in the training dataset.

#We want to drop it for our analysisso we will assign it a variable to recall later 

row_id = test.row_id
test.drop(columns=['row_id'], axis=1, inplace=True)
train.describe()
test.describe()
print("Percent equal to zero:", (train[train.meter_reading == 0].shape[0] / len(train.meter_reading))*100)
# An example of meter readings for building_id 2

train[(train.building_id==2) & train.meter_reading>0].iloc[:100]
print("Percent above mean:", (train[train.meter_reading > train.meter_reading.mean()].shape[0] / train.shape[0]) * 100)
train.groupby('building_id').meter_reading.mean().sort_values()
train.isnull().sum() * 100 / len(test)
test.isnull().sum() * 100 / len(test)
#create a sample dataframe that filters our values greater than 5000 and 

# not equal to 0, so that our vizualizations are to scale

sample = train[(train.meter_reading < 5000)&(train.meter_reading > 0)] 

sample.sample(n=1000, random_state=1).hist(bins=50, figsize=(20,15)) 
# output a scatter matrix to see correlation between variables

# meter reading is the target variable

# the independent variables differ between each

from pandas.plotting import scatter_matrix

attributes = ['meter_reading', 'building_id', 'meter', 'site_id', 'primary_use']

scatter_matrix(sample[attributes].sample(n=100, random_state=1), figsize=(12, 8))
attributes = ['meter_reading','square_feet', 'year_built', 'floor_count']

scatter_matrix(sample[attributes].sample(n=100, random_state=1), figsize=(12, 8))
attributes = ['meter_reading', 'air_temperature', 'cloud_coverage', 'dew_temperature']

scatter_matrix(sample[attributes].sample(n=100, random_state=1), figsize=(12, 8))
attributes = ['meter_reading', 'sea_level_pressure', 'wind_direction', 'wind_speed']

scatter_matrix(sample[attributes].sample(n=100, random_state=1), figsize=(12, 8))
# the distribution of a sample from the square-feet column

s = train['square_feet'].dropna(axis=0)

s = s.sample(n=1000000, random_state=1)

sns.distplot(a=s, kde=True)
# the distribution of the data behind the illustration cubed

s = np.cbrt(s)

sns.distplot(a=s, kde=True)
#transform column by cubing all values

train.square_feet = np.cbrt(train.square_feet)

test.square_feet = np.cbrt(test.square_feet)
train['sea_temp'] = np.sqrt(np.square(train['air_temperature'] / train.sea_level_pressure * 100))

test['sea_temp'] = np.sqrt(np.square(test['air_temperature'] / test.sea_level_pressure * 100))
train['internal'] = np.square(train.air_temperature - 22)

test['internal'] = np.square(test.air_temperature - 22)
train['air_sq'] = np.square(train.internal / train.square_feet)

test['air_sq'] = np.square(test.internal / test.square_feet)
train.columns
# update the sample variable to include the new features

sample = train[(train.meter_reading < 5000)&(train.meter_reading > 0)] 





attributes = ['meter_reading', 'sea_temp', 'air_sq']

scatter_matrix(sample[attributes].sample(n=10000, random_state=1), figsize=(12, 8))
drop_train = ['floor_count', 'timestamp', 'year_built', 'cloud_coverage', 'precip_depth_1_hr']

drop_test = ['floor_count', 'timestamp', 'year_built', 'cloud_coverage', 'precip_depth_1_hr']



train.drop(drop_train, axis=1, inplace=True)

test.drop(drop_test, axis=1, inplace=True)
print(train.shape)

print(test.shape)
#X = train.drop(columns=['meter_reading'])

#y = train['meter_reading']
#from sklearn.model_selection import train_test_split, GridSearchCV

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1)
#categorical_features = ["building_id", "site_id", "meter", "primary_use", "hour", "weekday"]



#lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)



#lgb_test = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_features, free_raw_data=False)



params = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 30,

    "learning_rate": 0.1,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse"

}



#model = lgb.train(params, train_set=lgb_train,  num_boost_round=1000, valid_sets=[lgb_train, lgb_test], verbose_eval=200, early_stopping_rounds=200)



#predictions = model.predict(X_test, num_iteration=model.best_iteration)
#lgb_train_full = lgb.Dataset(X, label=y, categorical_feature=categorical_features, free_raw_data=False)



params = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 40,

    "learning_rate": 0.1,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse"

}



#model2 = lgb.train(params, train_set=lgb_train_full,  num_boost_round=2000)



#predictions = model2.predict(test, num_iteration=model.best_iteration)
#submission = pd.DataFrame({'row_id':row_id, 'meter_reading':predictions})

#submission.loc[submission.meter_reading < 0, 'meter_reading'] = 0

#submission.to_csv('/Users/DataScience/energy/lgbm4.csv', index=False)
#pd.read_csv('/Users/DataScience/energy/lgbm2.csv').head()
#submission.head(10)