import numpy as np

import pandas as pd

import gc

import lightgbm as lgb



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
%%time



metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}

weather_dtype = {"site_id":"uint8",'air_temperature':"float16",'cloud_coverage':"float16",'dew_temperature':"float16",'precip_depth_1_hr':"float16",

                 'sea_level_pressure':"float32",'wind_direction':"float16",'wind_speed':"float16"}

train_dtype = {'meter':"uint8",'building_id':'uint16'}



weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv", parse_dates=['timestamp'], dtype=weather_dtype)

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=['timestamp'], dtype=weather_dtype)

metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv", parse_dates=['timestamp'], dtype=train_dtype)

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv", parse_dates=['timestamp'], usecols=['building_id','meter','timestamp'], dtype=train_dtype)


#from sklearn.utils import resample

#train_size = 1_000_000

#test_size = 1_000_000

#seed = 947

#train = resample(train, replace=False, n_samples=train_size, random_state=seed)

#test = resample(test, replace=False, n_samples=test_size, random_state=seed)
train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)



metadata["floor_count"].fillna(int(metadata["floor_count"].mean()), inplace=True)



for df in [train, test]:

    df['Month'] = df['timestamp'].dt.month.astype("uint8")

    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")

    

train['meter_reading'] = np.log1p(train['meter_reading'])



metadata['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",

                                "Retail":"Other","Services":"Other","Technology/science":"Other","Food sales and service":"Other",

                                "Utility":"Other","Religious worship":"Other"},inplace=True)

metadata['square_feet'] = np.log1p(metadata['square_feet'])

metadata['year_built'].fillna(int(metadata["year_built"].mean()), inplace=True)

metadata['year_built'] = metadata['year_built'].astype('int16')
%%time

train = pd.merge(train,metadata,on='building_id',how='left')

test  = pd.merge(test,metadata,on='building_id',how='left')

gc.collect()

train = pd.merge(train,weather_train,on=['site_id','timestamp'],how='left')

test  = pd.merge(test,weather_test,on=['site_id','timestamp'],how='left')

gc.collect()
# Save space

for df in [train,test]:

    df['square_feet'] = df['square_feet'].astype('float16')

    

# Fill NA

cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_speed', "wind_direction"]

for col in cols:

    train[col].fillna(train[col].mean(),inplace=True)

    test[col].fillna(test[col].mean(),inplace=True)

    

# Drop nonsense entries

# As per the discussion in the following thread, https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083, there is some discrepancy in the meter_readings for different ste_id's and buildings. It makes sense to delete them

idx_to_drop = list((train[(train['site_id'] == 0) & (train['timestamp'] < "2016-05-21 00:00:00")]).index)

train.drop(idx_to_drop,axis='rows',inplace=True)



# dropping all the electricity meter readings that are 0, after considering them as anomalies.

idx_to_drop = list(train[(train['meter'] == "Electricity") & (train['meter_reading'] == 0)].index)

train.drop(idx_to_drop,axis='rows',inplace=True)
%%time



mean_temperature_per_site = train.groupby("site_id")["air_temperature"].mean()

train["mean_temperature_per_site"] = train["site_id"].map(mean_temperature_per_site)

test["mean_temperature_per_site"] = test["site_id"].map(mean_temperature_per_site)



number_unique_meter_per_building = train.groupby('building_id')['meter'].nunique()

train['number_unique_meter_per_building'] = train['building_id'].map(number_unique_meter_per_building)



mean_meter_reading_per_building = train.groupby('building_id')['meter_reading'].mean()

train['mean_meter_reading_per_building'] = train['building_id'].map(mean_meter_reading_per_building)

median_meter_reading_per_building = train.groupby('building_id')['meter_reading'].median()

train['median_meter_reading_per_building'] = train['building_id'].map(median_meter_reading_per_building)

std_meter_reading_per_building = train.groupby('building_id')['meter_reading'].std()

train['std_meter_reading_per_building'] = train['building_id'].map(std_meter_reading_per_building)



mean_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].mean()

train['mean_meter_reading_on_year_built'] = train['year_built'].map(mean_meter_reading_on_year_built)

median_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].median()

train['median_meter_reading_on_year_built'] = train['year_built'].map(median_meter_reading_on_year_built)

std_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].std()

train['std_meter_reading_on_year_built'] = train['year_built'].map(std_meter_reading_on_year_built)



mean_meter_reading_per_meter = train.groupby('meter')['meter_reading'].mean()

train['mean_meter_reading_per_meter'] = train['meter'].map(mean_meter_reading_per_meter)

median_meter_reading_per_meter = train.groupby('meter')['meter_reading'].median()

train['median_meter_reading_per_meter'] = train['meter'].map(median_meter_reading_per_meter)

std_meter_reading_per_meter = train.groupby('meter')['meter_reading'].std()

train['std_meter_reading_per_meter'] = train['meter'].map(std_meter_reading_per_meter)



mean_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].mean()

train['mean_meter_reading_per_primary_usage'] = train['primary_use'].map(mean_meter_reading_per_primary_usage)

median_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].median()

train['median_meter_reading_per_primary_usage'] = train['primary_use'].map(median_meter_reading_per_primary_usage)

std_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].std()

train['std_meter_reading_per_primary_usage'] = train['primary_use'].map(std_meter_reading_per_primary_usage)



mean_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].mean()

train['mean_meter_reading_per_site_id'] = train['site_id'].map(mean_meter_reading_per_site_id)

median_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].median()

train['median_meter_reading_per_site_id'] = train['site_id'].map(median_meter_reading_per_site_id)

std_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].std()

train['std_meter_reading_per_site_id'] = train['site_id'].map(std_meter_reading_per_site_id)





test['number_unique_meter_per_building'] = test['building_id'].map(number_unique_meter_per_building)



test['mean_meter_reading_per_building'] = test['building_id'].map(mean_meter_reading_per_building)

test['median_meter_reading_per_building'] = test['building_id'].map(median_meter_reading_per_building)

test['std_meter_reading_per_building'] = test['building_id'].map(std_meter_reading_per_building)



test['mean_meter_reading_on_year_built'] = test['year_built'].map(mean_meter_reading_on_year_built)

test['median_meter_reading_on_year_built'] = test['year_built'].map(median_meter_reading_on_year_built)

test['std_meter_reading_on_year_built'] = test['year_built'].map(std_meter_reading_on_year_built)



test['mean_meter_reading_per_meter'] = test['meter'].map(mean_meter_reading_per_meter)

test['median_meter_reading_per_meter'] = test['meter'].map(median_meter_reading_per_meter)

test['std_meter_reading_per_meter'] = test['meter'].map(std_meter_reading_per_meter)



test['mean_meter_reading_per_primary_usage'] = test['primary_use'].map(mean_meter_reading_per_primary_usage)

test['median_meter_reading_per_primary_usage'] = test['primary_use'].map(median_meter_reading_per_primary_usage)

test['std_meter_reading_per_primary_usage'] = test['primary_use'].map(std_meter_reading_per_primary_usage)



test['mean_meter_reading_per_site_id'] = test['site_id'].map(mean_meter_reading_per_site_id)

test['median_meter_reading_per_site_id'] = test['site_id'].map(median_meter_reading_per_site_id)

test['std_meter_reading_per_site_id'] = test['site_id'].map(std_meter_reading_per_site_id)
%%time

for df in [train, test]:

    df['mean_temperature_per_site'] = df['mean_temperature_per_site'].astype("float16")

    

    df['mean_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")

    df['median_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")

    df['std_meter_reading_per_building'] = df['std_meter_reading_per_building'].astype("float16")

    

    df['mean_meter_reading_on_year_built'] = df['mean_meter_reading_on_year_built'].astype("float16")

    df['median_meter_reading_on_year_built'] = df['median_meter_reading_on_year_built'].astype("float16")

    df['std_meter_reading_on_year_built'] = df['std_meter_reading_on_year_built'].astype("float16")

    

    df['mean_meter_reading_per_meter'] = df['mean_meter_reading_per_meter'].astype("float16")

    df['median_meter_reading_per_meter'] = df['median_meter_reading_per_meter'].astype("float16")

    df['std_meter_reading_per_meter'] = df['std_meter_reading_per_meter'].astype("float16")

    

    df['mean_meter_reading_per_primary_usage'] = df['mean_meter_reading_per_primary_usage'].astype("float16")

    df['median_meter_reading_per_primary_usage'] = df['median_meter_reading_per_primary_usage'].astype("float16")

    df['std_meter_reading_per_primary_usage'] = df['std_meter_reading_per_primary_usage'].astype("float16")

    

    df['mean_meter_reading_per_site_id'] = df['mean_meter_reading_per_site_id'].astype("float16")

    df['median_meter_reading_per_site_id'] = df['median_meter_reading_per_site_id'].astype("float16")

    df['std_meter_reading_per_site_id'] = df['std_meter_reading_per_site_id'].astype("float16")

    

    df['number_unique_meter_per_building'] = df['number_unique_meter_per_building'].astype('uint8')

gc.collect()
train.drop('timestamp',axis=1,inplace=True)

test.drop('timestamp',axis=1,inplace=True)



le = LabelEncoder()



train['meter']= le.fit_transform(train['meter']).astype("uint8")

test['meter']= le.fit_transform(test['meter']).astype("uint8")

train['primary_use']= le.fit_transform(train['primary_use']).astype("uint8")

test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")



print (train.shape, test.shape)
%%time

# Let's check the correlation between the variables and eliminate the one's that have high correlation

# Threshold for removing correlated variables

threshold = 0.90



# Absolute value correlation matrix

corr_matrix = train.corr().abs()

# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]



print('There are %d columns to remove.' % (len(to_drop)))

print ("Following columns can be dropped {}".format(to_drop))



train.drop(to_drop,axis=1,inplace=True)

test.drop(to_drop,axis=1,inplace=True)



y = train['meter_reading']

train.drop('meter_reading',axis=1,inplace=True)

categorical_cols = ['building_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth']
x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.05,random_state=573)

print (x_train.shape)

print (x_test.shape)
%%time



lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_cols)

lgb_test = lgb.Dataset(x_test, y_test, categorical_feature=categorical_cols)

del x_train, x_test , y_train, y_test



params = {'feature_fraction': 0.8,

          'bagging_fraction': 0.7,

          'objective': 'regression',

          'max_depth': -1,

          'learning_rate': 0.1,

          "boosting_type": "gbdt",

          "bagging_seed": 321,

          "metric": 'rmse',

          "verbosity": -1,

          'reg_alpha': 1,

          'reg_lambda': 2,

          'random_state': 123,

          'num_leaves': 70

         }



reg = lgb.train(params, lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval=100)
%%time



del train, y, lgb_train

predictions = []

step = 50000

for i in range(0, len(test), step):

    predictions.extend(np.expm1(reg.predict(test.iloc[i: min(i+step, len(test)), :], num_iteration=reg.best_iteration)))
%%time



Submission = pd.DataFrame(test.index,columns=['row_id'])

Submission['meter_reading'] = predictions

Submission['meter_reading'].clip(lower=0,upper=None,inplace=True)

Submission.to_csv("lgbm.csv",index=None)