import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import gc, math



from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
sns.set(rc={'figure.figsize':(11,8)})

sns.set(style="whitegrid")
%%time

metadata_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv', parse_dates=['timestamp'])

test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv', parse_dates=['timestamp'])

weather_train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv', parse_dates=['timestamp'])

weather_test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv', parse_dates=['timestamp'])
train_df.head()
metadata_df.head()
weather_train_df.shape
weather_train_df.head()
test_df.head()
weather = pd.concat([weather_train_df,weather_test_df],ignore_index=True)

weather_key = ['site_id', 'timestamp']



temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()



# calculate ranks of hourly temperatures within date/site_id chunks

temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')



# create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)

df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)



# Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.

site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)

site_ids_offsets.index.name = 'site_id'



def timestamp_align(df):

    df['offset'] = df.site_id.map(site_ids_offsets)

    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))

    df['timestamp'] = df['timestamp_aligned']

    del df['timestamp_aligned']

    return df
weather_train_df.tail()
weather_train_df = timestamp_align(weather_train_df)

weather_test_df = timestamp_align(weather_test_df)
del weather 

del df_2d

del temp_skeleton

del site_ids_offsets
weather_train_df.tail()
def add_lag_feature(weather_df, window=3):

    group_df = weather_df.groupby('site_id')

    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr']

    rolled = group_df[cols].rolling(window=window, min_periods=0)

    lag_mean = rolled.mean().reset_index().astype(np.float16)

    lag_std = rolled.std().reset_index().astype(np.float16)

    for col in cols:

        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
add_lag_feature(weather_train_df, window=72)

add_lag_feature(weather_test_df, window=72)
weather_train_df.columns
weather_train_df.isna().sum()
weather_test_df.isna().sum()
weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
weather_train_df.isna().sum()
weather_test_df.isna().sum()
train_df['meter_reading'] = np.log1p(train_df['meter_reading'])
weather_train_df.head()
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
le = LabelEncoder()

metadata_df['primary_use'] = le.fit_transform(metadata_df['primary_use'])
metadata_df = reduce_mem_usage(metadata_df)

train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)

weather_train_df = reduce_mem_usage(weather_train_df)

weather_test_df = reduce_mem_usage(weather_test_df)
print (f'Training data shape: {train_df.shape}')

print (f'Weather training shape: {weather_train_df.shape}')

print (f'Weather training shape: {weather_test_df.shape}')

print (f'Weather testing shape: {metadata_df.shape}')

print (f'Test data shape: {test_df.shape}')
train_df.head()
weather_train_df.head()
metadata_df.head()
test_df.head()
train_df.head()
%%time

full_train_df = train_df.merge(metadata_df, on='building_id', how='left')

full_train_df = full_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
full_train_df = full_train_df.loc[~(full_train_df['air_temperature'].isnull() & full_train_df['cloud_coverage'].isnull() & full_train_df['dew_temperature'].isnull() & full_train_df['precip_depth_1_hr'].isnull() & full_train_df['sea_level_pressure'].isnull() & full_train_df['wind_direction'].isnull() & full_train_df['wind_speed'].isnull() & full_train_df['offset'].isnull())]
full_train_df.shape
# Delete unnecessary dataframes to decrease memory usage

del train_df

del weather_train_df

gc.collect()
%%time

full_test_df = test_df.merge(metadata_df, on='building_id', how='left')

full_test_df = full_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
full_test_df.shape
# Delete unnecessary dataframes to decrease memory usage

del metadata_df

del weather_test_df

del test_df

gc.collect()
ax = sns.barplot(pd.unique(full_train_df['primary_use']), full_train_df['primary_use'].value_counts())

ax.set(xlabel='Primary Usage', ylabel='# of records', title='Primary Usage vs. # of records')

ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")

plt.show()
meter_types = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}

ax = sns.barplot(np.vectorize(meter_types.get)(pd.unique(full_train_df['meter'])), full_train_df['meter'].value_counts())

ax.set(xlabel='Meter Type', ylabel='# of records', title='Meter type vs. # of records')

plt.show()
# Average meter reading

print (f'Average meter reading: {full_train_df.meter_reading.mean()} kWh')
ax = sns.barplot(np.vectorize(meter_types.get)(full_train_df.groupby(['meter'])['meter_reading'].mean().keys()), full_train_df.groupby(['meter'])['meter_reading'].mean())

ax.set(xlabel='Meter Type', ylabel='Meter reading', title='Meter type vs. Meter Reading')

plt.show()
fig, ax = plt.subplots(1,1,figsize=(14, 6))

ax.set(xlabel='Year Built', ylabel='# Of Buildings', title='Buildings built in each year')

full_train_df['year_built'].value_counts(dropna=False).sort_index().plot(ax=ax)

full_test_df['year_built'].value_counts(dropna=False).sort_index().plot(ax=ax)

ax.legend(['Train', 'Test']);
fig, ax = plt.subplots(1,1,figsize=(15, 7))

full_train_df.groupby(['building_id'])['square_feet'].mean().plot(ax=ax)

ax.set(xlabel='Building ID', ylabel='Area in Square Feet', title='Square Feet area of buildings')

plt.show()
pd.DataFrame(full_train_df.isna().sum().sort_values(ascending=False), columns=['NaN Count'])
def mean_without_overflow_fast(col):

    col /= len(col)

    return col.mean() * len(col)
missing_values = (100-full_train_df.count() / len(full_train_df) * 100).sort_values(ascending=False)
%%time

missing_features = full_train_df.loc[:, missing_values > 0.0]

missing_features = missing_features.apply(mean_without_overflow_fast)
for key in full_train_df.loc[:, missing_values > 0.0].keys():

    if key == 'year_built' or key == 'floor_count':

        full_train_df[key].fillna(math.floor(missing_features[key]), inplace=True)

        full_test_df[key].fillna(math.floor(missing_features[key]), inplace=True)

    else:

        full_train_df[key].fillna(missing_features[key], inplace=True)

        full_test_df[key].fillna(missing_features[key], inplace=True)
full_train_df.tail()
full_test_df.tail()
full_train_df['timestamp'].dtype
full_train_df["timestamp"] = pd.to_datetime(full_train_df["timestamp"])

full_test_df["timestamp"] = pd.to_datetime(full_test_df["timestamp"])
def transform(df):

    df['hour'] = np.uint8(df['timestamp'].dt.hour)

    df['day'] = np.uint8(df['timestamp'].dt.day)

    df['weekday'] = np.uint8(df['timestamp'].dt.weekday)

    df['month'] = np.uint8(df['timestamp'].dt.month)

    df['year'] = np.uint8(df['timestamp'].dt.year-1900)

    

    df['square_feet'] = np.log(df['square_feet'])

    

    return df
full_train_df = transform(full_train_df)

full_test_df = transform(full_test_df)
dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')

us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

full_train_df['is_holiday'] = (full_train_df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

full_test_df['is_holiday'] = (full_test_df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)
# Assuming 5 days a week for all the given buildings

full_train_df.loc[(full_train_df['weekday'] == 5) | (full_train_df['weekday'] == 6) , 'is_holiday'] = 1

full_test_df.loc[(full_test_df['weekday']) == 5 | (full_test_df['weekday'] == 6) , 'is_holiday'] = 1
full_train_df.shape
full_train_df = full_train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
full_train_df.shape
full_test_df = full_test_df.drop(['timestamp'], axis=1)

full_train_df = full_train_df.drop(['timestamp'], axis=1)

print (f'Shape of training dataset: {full_train_df.shape}')

print (f'Shape of testing dataset: {full_test_df.shape}')
full_train_df.tail()
full_train_df.tail()
## Reducing memory

full_train_df = reduce_mem_usage(full_train_df)

full_test_df = reduce_mem_usage(full_test_df)

gc.collect()
# def degToCompass(num):

#     val=int((num/22.5)+.5)

#     arr=[i for i in range(0,16)]

#     return arr[(val % 16)]
# full_train_df['wind_direction'] = full_train_df['wind_direction'].apply(degToCompass)
# beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 

#           (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]



# for item in beaufort:

#     full_train_df.loc[(full_train_df['wind_speed']>=item[1]) & (full_train_df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]
# le = LabelEncoder()

# full_train_df['primary_use'] = le.fit_transform(full_train_df['primary_use'])



categoricals = ['site_id', 'building_id', 'primary_use', 'hour', 'weekday', 'meter',  'wind_direction', 'is_holiday']

# drop_cols = ['sea_level_pressure', 'wind_speed']

numericals = ['square_feet', 'year_built', 'air_temperature', 'cloud_coverage',

              'dew_temperature', 'precip_depth_1_hr', 'floor_count', 'air_temperature_mean_lag72',

       'cloud_coverage_mean_lag72', 'dew_temperature_mean_lag72',

       'precip_depth_1_hr_mean_lag72']



feat_cols = categoricals + numericals
full_train_df.tail()
full_train_df = reduce_mem_usage(full_train_df)

gc.collect()
target = full_train_df["meter_reading"]

del full_train_df["meter_reading"]
# full_train_df.drop(drop_cols, axis=1)

# gc.collect()
# Save the testing dataset to freeup the RAM. We'll read after training

full_test_df.to_pickle('full_test_df.pkl')

del full_test_df

gc.collect()
params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

            'subsample': 0.4,

            'subsample_freq': 1,

            'learning_rate': 0.3,

            'num_leaves': 40,

            'feature_fraction': 0.80,

            'lambda_l1': 1,

            'lambda_l2': 1

            }



folds = 2

seed = 666



kf = StratifiedKFold(n_splits=folds, shuffle=False, random_state=seed)



models = []

for train_index, val_index in kf.split(full_train_df, full_train_df['building_id']):

    train_X = full_train_df[feat_cols].iloc[train_index]

    val_X = full_train_df[feat_cols].iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=(lgb_train, lgb_eval),

                early_stopping_rounds=100,

                verbose_eval = 100)

    models.append(gbm)
del full_train_df, train_X, val_X, lgb_train, lgb_eval, train_y, val_y, target

gc.collect()
full_test_df = pd.read_pickle('full_test_df.pkl')
# full_test_df['wind_direction'] = full_test_df['wind_direction'].apply(degToCompass)
# for item in beaufort:

#     full_test_df.loc[(full_test_df['wind_speed']>=item[1]) & (full_test_df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]
full_test_df = full_test_df[feat_cols]
i=0

res=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(full_test_df.shape[0]/50000)))):

    res.append(np.expm1(sum([model.predict(full_test_df.iloc[i:i+step_size]) for model in models])/folds))

    i+=step_size
res = np.concatenate(res)
submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

submission['meter_reading'] = res

submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0

submission.to_csv('submission_fe_lgbm.csv', index=False)

submission