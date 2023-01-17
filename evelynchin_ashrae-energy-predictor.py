import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline

warnings.filterwarnings('ignore')



import gc

import matplotlib.patches as patches

from scipy import stats

from scipy.stats import skew



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 100)



py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import os,random, math, psutil, pickle

from sklearn.preprocessing import LabelEncoder

import missingno as msno



from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# DATA PROCESSING



# 1. Load Data

# 2. Reduce Memory

# 3. Average Imputation

# 4. Conversions

    # Convert Timestamps

    # Convert Strings to Categorical

    # Convert Primary Use through Label Encoder

    # Convert Wind Direction Degrees to 16-Wind Compass

# 5. Merge Metadata, Weather, and Train/Test DF

# 6. Data Minification

# 7. Create Features

    # Create Beaufort Wind Scale

    # Create Age Feature

    # Create Datetime Features

# 8. Assign DTypes
# Load Data

root = '../input/ashrae-energy-prediction/'

train_df = pd.read_csv(root + 'train.csv')

weather_train_df = pd.read_csv(root + 'weather_train.csv')

test_df = pd.read_csv(root + 'test.csv')

weather_test_df = pd.read_csv(root + 'weather_test.csv')

building_meta_df = pd.read_csv(root + 'building_metadata.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')



print('Size of train_df data', train_df.shape)

print('Size of weather_train_df data', weather_train_df.shape)

print('Size of weather_test_df data', weather_test_df.shape)

print('Size of building_meta_df data', building_meta_df.shape)
# Reduce Memory

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



train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)

weather_train_df = reduce_mem_usage(weather_train_df)

weather_test_df = reduce_mem_usage(weather_test_df)

building_meta_df = reduce_mem_usage(building_meta_df)
# Merge Metadata, Weather, and Train/Test DF

temp_df = train_df[['building_id']]

temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')

del temp_df['building_id']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['building_id']]

temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')

del temp_df['building_id']

test_df = pd.concat([test_df, temp_df], axis=1)

del temp_df, building_meta_df



temp_df = train_df[['site_id','timestamp']]

temp_df = temp_df.merge(weather_train_df, on=['site_id','timestamp'], how='left')

del temp_df['site_id'], temp_df['timestamp']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['site_id','timestamp']]

temp_df = temp_df.merge(weather_test_df, on=['site_id','timestamp'], how='left')

del temp_df['site_id'], temp_df['timestamp']

test_df = pd.concat([test_df, temp_df], axis=1)

del temp_df, weather_train_df, weather_test_df
# Average Imputation

def average_imputation(df, column_name):

    imputation = df.groupby(['timestamp'])[column_name].mean()

    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)

    del imputation

    return df



train_df = average_imputation(train_df, 'wind_speed')

train_df = average_imputation(train_df, 'wind_direction')

test_df = average_imputation(test_df, 'wind_speed')

test_df = average_imputation(test_df, 'wind_direction')
# Conversions



# Convert Timestamps

train_df['timestamp'] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])



# Convert Strings to Categorical

train_df['primary_use'] = train_df['primary_use'].astype('category')

test_df['primary_use'] = test_df['primary_use'].astype('category')



# Convert Primary Use through Label Encoder

le = LabelEncoder()

train_df['primary_use'] = le.fit_transform(train_df['primary_use']).astype(np.int8)

test_df['primary_use'] = le.fit_transform(test_df['primary_use']).astype(np.int8)



# Convert Wind Direction Degrees to 16-Wind Compass

def degToCompass(num):

    val = int((num / 22.5))

    arr = [i for i in range(0, 16)]

    return arr[(val % 16)]



train_df['wind_direction'] = train_df['wind_direction'].apply(degToCompass)

test_df['wind_direction'] = test_df['wind_direction'].apply(degToCompass)

train_df.head()
# Data Minification

train_df.to_pickle('train_df.pkl')

test_df.to_pickle('test_df.pkl')

del train_df, test_df

gc.collect() # release unreferenced memory

train_df = pd.read_pickle('train_df.pkl')

test_df = pd.read_pickle('test_df.pkl')
# Create Features



# Create Beaufort Wind Scale Variable

beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 

            (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]



for item in beaufort:

    train_df.loc[(train_df['wind_speed'] >= item[1]) & (train_df['wind_speed'] < item[2]), 'beaufort_scale'] = item[0]

    test_df.loc[(test_df['wind_speed'] >= item[1]) & (test_df['wind_speed'] < item[2]), 'beaufort_scale'] = item[0]

    

# Create Age Feature

train_df['age'] = train_df['year_built'].max() - train_df['year_built'] + 1

test_df['age'] = test_df['year_built'].max() - test_df['year_built'] + 1



# Create Datetime Features

train_df['month_datetime'] = train_df['timestamp'].dt.month.astype(np.int8)

train_df['weekofyear_datetime'] = train_df['timestamp'].dt.weekofyear.astype(np.int8)

train_df['dayofyear_datetime'] = train_df['timestamp'].dt.dayofyear.astype(np.int16)

train_df['hour_datetime'] = train_df['timestamp'].dt.hour.astype(np.int8)

train_df['day_week'] = train_df['timestamp'].dt.dayofweek.astype(np.int8)

train_df['day_month_datetime'] = train_df['timestamp'].dt.day.astype(np.int8)

train_df['week_month_datetime'] = train_df['timestamp'].dt.day/7

train_df['week_month_datetime'] = train_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)



test_df['month_datetime'] = test_df['timestamp'].dt.month.astype(np.int8)

test_df['weekofyear_datetime'] = test_df['timestamp'].dt.weekofyear.astype(np.int8)

test_df['dayofyear_datetime'] = test_df['timestamp'].dt.dayofyear.astype(np.int16)    

test_df['hour_datetime'] = test_df['timestamp'].dt.hour.astype(np.int8)

test_df['day_week'] = test_df['timestamp'].dt.dayofweek.astype(np.int8)

test_df['day_month_datetime'] = test_df['timestamp'].dt.day.astype(np.int8)

test_df['week_month_datetime'] = test_df['timestamp'].dt.day/7

test_df['week_month_datetime'] = test_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)
# Assign DTypes

train_df['beaufort_scale'] = train_df['beaufort_scale'].astype(np.uint8)

train_df["wind_direction"] = train_df['wind_direction'].astype(np.uint8)

train_df["meter"] = train_df['meter'].astype(np.uint8)

train_df["site_id"] = train_df['site_id'].astype(np.uint8)



test_df['beaufort_scale'] = test_df['beaufort_scale'].astype(np.uint8)

test_df["wind_direction"] = test_df['wind_direction'].astype(np.uint8)

test_df["meter"] = test_df['meter'].astype(np.uint8)

test_df["site_id"] = test_df['site_id'].astype(np.uint8)
# EXPLORATORY DATA ANALYSIS



# 1. Target Distribution

# 2. Handle Missing Values

    # Visualize Missing Values

    # Fill In Missing Values with -999

# 3. Outlier Analysis

# 4. Drop Columns
# Target Distribution

plt.figure(figsize = (15,5))

train_df['meter_reading'].plot()

train_df.meter_reading.describe()
# Handle Missing Values



# Visualize Missing Values

total = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)

missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

msno.matrix(train_df.head(20000), figsize=(15, 5))

missing_train_data.head(10)
# Fill In Missing Values with -999

train_df['floor_count'] = train_df['floor_count'].fillna(-999).astype(np.int16)

test_df['floor_count'] = test_df['floor_count'].fillna(-999).astype(np.int16)

train_df['year_built'] = train_df['year_built'].fillna(-999).astype(np.int16)

test_df['year_built'] = test_df['year_built'].fillna(-999).astype(np.int16)

train_df['age'] = train_df['age'].fillna(-999).astype(np.int16)

test_df['age'] = test_df['age'].fillna(-999).astype(np.int16)

train_df['cloud_coverage'] = train_df['cloud_coverage'].fillna(-999).astype(np.int16)

test_df['cloud_coverage'] = test_df['cloud_coverage'].fillna(-999).astype(np.int16)
# Outlier Analysis

energy_types_dict = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}

energy_types = ['electricity', 'chilledwater', 'steam', 'hotwater']



plt.figure(figsize=(13, 3))

tmp_df = train_df.meter.value_counts()

tmp_df.index = energy_types

tmp_df.sort_values().plot(kind="barh")

plt.title("Most readings measure electricity")

plt.xlabel("Number of measurements")

plt.ylabel("Meter Type")

plt.tight_layout()

plt.show()



plt.figure(figsize=(13, 3))

tmp_df = train_df.groupby("meter").meter_reading.sum()

tmp_df.index = energy_types

tmp_df.sort_values().plot(kind="barh")

plt.title("Steam generation consumes the most energy")

plt.xlabel("Amount of consumed energy")

plt.ylabel("Meter Type")

plt.tight_layout()

plt.show()



plt.figure(figsize=(13, 3))

sns.distplot(train_df.meter_reading, hist=False)

plt.title("Target variable (meter_reading) is highly skewed")

plt.ylabel("Number of measurements")

plt.xlabel("Amount of consumed energy")

plt.xlim(0, train_df.meter_reading.max() + 100_000)

plt.tight_layout()

plt.show()



plt.figure(figsize=(13, 3))

sns.distplot(np.log1p(train_df.meter_reading))

plt.title("Log Transform: Better but still skewed")

plt.ylabel("Number of measurements")

plt.xlabel("Amount of consumed energy")

plt.xlim(0, 12)

plt.tight_layout()

plt.show()



plt.figure(figsize=(13, 3))

for idx in range(0,4):

    sns.distplot(np.log1p(train_df[train_df.meter == idx].meter_reading), hist=False, label=energy_types[idx])

plt.title("Log Transform: energy type distributions look relatively skewed")

plt.ylabel("Number of measurements")

plt.xlabel("Amount of consumed energy")

plt.legend()

plt.xlim(0, 12)

plt.tight_layout()

plt.show()
temp_df = train_df.groupby("primary_use").meter_reading.sum().sort_values()



plt.figure(figsize=(13,4))

temp_df.plot(kind="barh")

plt.title("Education buildings consume by far most of energy")

plt.xlabel("Number of measurements")

plt.ylabel("Primary use")

plt.tight_layout()

plt.show()



plt.figure(figsize=(13,4))

temp_df[:-1].plot(kind="barh")

plt.title("Among other types, office buildings consume most energy")

plt.xlabel("Number of measurements")

plt.ylabel("Primary use w/o «Education»")

plt.tight_layout()

plt.show()
# Drop Columns

del train_df["timestamp"], test_df["timestamp"]



categoricals = ["site_id", "building_id", "primary_use",  "meter",  "wind_direction"]

numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage", 

              "dew_temperature", 'precip_depth_1_hr', 'floor_count', 'beaufort_scale']

drop_cols = ["sea_level_pressure", "wind_speed"]

feat_cols = categoricals + numericals



target = np.log1p(train_df["meter_reading"])

del train_df["meter_reading"] 

train_df = train_df.drop(drop_cols, axis = 1)
# MODEL



# 1. Model
# Model

params = {'boosting_type': 'gbdt', 'objective': 'regression',

          'metric': {'rmse'}, 'subsample_freq': 1, 'learning_rate': 0.3,

          'bagging_freq': 5, 'num_leaves': 330, 'feature_fraction': 0.9, 

          'lambda_l1': 1, 'lambda_l2': 1}



folds = 5

seed = 666

shuffle = False

kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)



models = []

for train_index, val_index in kf.split(train_df[feat_cols], train_df['building_id']):

    train_X = train_df[feat_cols].iloc[train_index]

    val_X = train_df[feat_cols].iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params, lgb_train, num_boost_round=500,

                    valid_sets=(lgb_train, lgb_eval),

                    early_stopping_rounds=50, verbose_eval = 50)

    models.append(gbm)
del train_df

gc.collect()



test_df = test_df[feat_cols]

i = 0

res = []

step_size = 50000

for j in tqdm(range(int(np.ceil(test_df.shape[0]/50000)))):

    res.append(np.expm1(sum([model.predict(test_df.iloc[i:i + step_size]) for model in models]) / folds))

    i += step_size

res = np.concatenate(res)
sample_submission['meter_reading'] = res

sample_submission.loc[sample_submission['meter_reading'] < 0, 'meter_reading'] = 0

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()