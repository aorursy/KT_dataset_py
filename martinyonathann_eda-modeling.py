import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from time import time

import datetime

import gc

pd.set_option('display.max_columns',100)

pd.set_option('display.float_format', lambda x: '%.5f' % x)

from sklearn.model_selection import train_test_split,KFold

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
metadata_type= {'site_id':"uint8",'building_id':"uint16",'square_feet':'float32',

               'year_built':'float32','floor_count':"float16"}

metadata=pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv",dtype=metadata_type)

metadata.info()
weather_dtype={"site_id":"uint8","air_temperature":"float16","cloud_coverage":"float16",

              "dew_temperature":"float16","precip_depth_1_hr":"float16","sea_level_pressure":"float32",

              "wind_direction":"float16","wind_speed":"float16"}

weather_train=pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",dtype=weather_dtype)

weather_test=pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",dtype=weather_dtype)

print(weather_train.info())

print("==================================================")

print(weather_test.info())
df_dtype={"meter":"uint8","building_id":"uint16"}

df_train=pd.read_csv("../input/ashrae-energy-prediction/train.csv",parse_dates=['timestamp'],dtype=df_dtype)

test_cols=['building_id','meter','timestamp']

df_test=pd.read_csv("../input/ashrae-energy-prediction/train.csv",usecols=test_cols,parse_dates=['timestamp'],dtype=df_dtype)
metadata.head()
weather_train.head()
weather_test.head()
df_train.head()
df_test.head()
metadata.isna().sum()/len(metadata)
#drop the missing colomn > 60 %

metadata.drop('floor_count',axis=1,inplace=True)
missing_weather=pd.DataFrame(weather_train.isna().sum()/len(weather_train),columns=["Weather_Train_Missing"])

missing_weather["Weather_Test_Missing"]=pd.DataFrame(weather_test.isna().sum()/len(weather_test))

missing_weather
missing_train_test = pd.DataFrame(df_train.isna().sum()/len(df_train),columns=["Missing_Pct_Train"])

missing_train_test["Missing_Pct_Test"] = df_test.isna().sum()/len(df_test)

missing_train_test
df_train.head()
df_train.describe(include='all')
df_train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

df_test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
sns.countplot(df_train['meter'])

plt.title("Distribution of Meter Id Code")

plt.xlabel("Meter Id Code")

plt.ylabel("Frequency")
print ("There are {} unique Buildings in the training data".format(df_train['building_id'].nunique()))
df_train['building_id'].value_counts(dropna=False).head(20)
df_train[df_train['building_id']==1094]['meter'].unique()
df_train.groupby('meter')['meter_reading'].agg(['min','max','mean','median',

                                               'count','std'])
for df in [df_train, df_test]:

    df['Month'] = df['timestamp'].dt.month.astype("uint8")

    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")
df_train.groupby(['meter','Month'])['meter_reading'].agg(['max','mean','median','count','std'])
df_train['meter_reading'].describe()
sns.distplot(np.log1p(df_train['meter_reading']),kde=False)

plt.title("Distribution of Log of Meter Reading Variable")
sns.boxplot(df_train[df_train['meter'] == "Electricity"]['meter_reading'])

plt.title("Boxplot of Meter Reading Variable for the Meter Type: Electricity")
sns.boxplot(df_train[df_train['meter'] == "ChilledWater"]['meter_reading'])

plt.title("Boxplot of Meter Reading Variable for the Meter Type: Electricity")

# Not many outliers here. 
sns.boxplot(df_train[df_train['meter'] == "HotWater"]['meter_reading'])

plt.title("Boxplot of Meter Reading Variable for the Meter Type: Electricity")

# We can see a single value that is way off from the rest. 
sns.boxplot(df_train[df_train['meter'] == "Steam"]['meter_reading'])

plt.title("Boxplot of Meter Reading Variable for the Meter Type: Electricity") 
df_train['meter_reading']=np.log1p(df_train['meter_reading'])
sns.distplot(df_train[df_train['meter'] == "Electricity"]['meter_reading'],kde=False)

plt.title("Distribution of Meter Reading per MeterID code: Electricity")
sns.distplot(df_train[df_train['meter'] == "ChilledWater"]['meter_reading'],kde=False)

plt.title("Distribution of Meter Reading per MeterID code: Chilledwater")
sns.distplot(df_train[df_train['meter'] == "Steam"]['meter_reading'],kde=False)

plt.title("Distribution of Meter Reading per MeterID code: Steam")
sns.distplot(df_train[df_train['meter'] == "HotWater"]['meter_reading'],kde=False)

plt.title("Distribution of Meter Reading per MeterID code: Hotwater")
metadata.info()
metadata.head()
cols = ['site_id','primary_use','building_id','year_built']

for col in cols:

    print ("Number of Unique Values in the {} column are:".format(col),metadata[col].nunique())
cols = ['site_id','primary_use','year_built']

for col in cols:

    print ("Unique Values in the {} column are:".format(col),metadata[col].unique())

    print ("\n")
sns.countplot(metadata['site_id'])

plt.title("Count of Site_id in the Metadata table")

plt.xlabel("Site_Id")

plt.ylabel("Count")
plt.figure(figsize=(8,6))

metadata['primary_use'].value_counts().sort_values().plot(kind='bar')

plt.title("Count of Primary_Use Variable in the Metadata table")

plt.xlabel("Primary Use")

plt.ylabel("Count")

plt.xticks(rotation=90)
metadata['primary_use'].value_counts(normalize=True)
metadata['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",

                                "Retail":"Other","Services":"Other","Technology/science":"Other","Food sales and service":"Other",

                                "Utility":"Other","Religious worship":"Other"},inplace=True)
metadata['square_feet'].describe()
sns.boxplot(metadata['square_feet'])
metadata['square_feet'] = np.log1p(metadata['square_feet'])
sns.distplot(metadata['square_feet'])

plt.title("Distribution of Square Feet variable of Metadata Table")

plt.xlabel("Area in Square Feet")

plt.ylabel("Frequency")

# Looks like a normal distribution distribution
sns.boxplot(metadata['square_feet'])

plt.title("Box Plot of Square Feet Variable")

# There are a few outliers visible
metadata.groupby('primary_use')['square_feet'].agg(['mean','median','count']).sort_values(by='count')

# Parking has the highest average are although the count is less.

# Education has the highest count as can be seen in the countplot above.
metadata['year_built'].value_counts().sort_values().plot(kind='bar',figsize=(15,6))

plt.xlabel("Year Built")

plt.ylabel("Count")

plt.title("Distribution of Year Built Variable")
metadata.groupby('primary_use')['square_feet'].agg(['count','mean','median']).sort_values(by='count')
metadata.head()
metadata['year_built'].fillna(-999, inplace=True)

metadata['year_built'] = metadata['year_built'].astype('int16')
weather_train.head()
cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

for col in cols:

    print (" Minimum Value of {} column is {}".format(col,weather_train[col].min()))

    print (" Maximum Value of {} column is {}".format(col,weather_train[col].max()))

    print ("----------------------------------------------------------------------")
weather_train.isna().sum()/len(weather_train)

weather_train[['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_speed']].describe()
weather_train['timestamp'].describe()
cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_speed']

for ind,col in enumerate(weather_train[cols]):

    plt.figure(ind)

    sns.distplot(weather_train[col].dropna())
cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_speed']

for ind,col in enumerate(weather_train[cols]):

    plt.figure(ind)

    sns.boxplot(weather_train[col].dropna())
weather_test.info(memory_usage='deep')
weather_test['timestamp'].describe()

weather_test['timestamp'] = weather_test['timestamp'].astype('datetime64')

# The time duration is similar to the test dataset.
%%time

df_train = pd.merge(df_train,metadata,on='building_id',how='left')

df_test  = pd.merge(df_test,metadata,on='building_id',how='left')

print ("Training Data Shape {}".format(df_train.shape))

print ("Testing Data Shape {}".format(df_test.shape))

gc.collect()
weather_train['timestamp'] = weather_train['timestamp'].astype('datetime64')

weather_train.info()
%%time

df_train = pd.merge(df_train,weather_train,on=['site_id','timestamp'],how='left')

df_test  = pd.merge(df_test,weather_test,on=['site_id','timestamp'],how='left')

print ("Training Data Shape {}".format(df_train.shape))

print ("Testing Data Shape {}".format(df_test.shape))

gc.collect()
for df in [df_train,df_test]:

    df['square_feet'] = df['square_feet'].astype('float16')
cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

for col in cols:

    df_train[col].fillna(df_train[col].mean(),inplace=True)

    df_test[col].fillna(df_test[col].mean(),inplace=True)
# As per the discussion in the following thread, https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083, there is some discrepancy in the meter_readings for different ste_id's and buildings. It makes sense to delete them

idx_to_drop = list((df_train[(df_train['site_id'] == 0) & (df_train['timestamp'] < "2016-05-21 00:00:00")]).index)

print (len(idx_to_drop))

df_train.drop(idx_to_drop,axis='rows',inplace=True)
%%time

number_unique_meter_per_building = df_train.groupby('building_id')['meter'].nunique()

df_train['number_unique_meter_per_building'] = df_train['building_id'].map(number_unique_meter_per_building)





mean_meter_reading_per_building = df_train.groupby('building_id')['meter_reading'].mean()

df_train['mean_meter_reading_per_building'] = df_train['building_id'].map(mean_meter_reading_per_building)

median_meter_reading_per_building = df_train.groupby('building_id')['meter_reading'].median()

df_train['median_meter_reading_per_building'] = df_train['building_id'].map(median_meter_reading_per_building)

std_meter_reading_per_building = df_train.groupby('building_id')['meter_reading'].std()

df_train['std_meter_reading_per_building'] = df_train['building_id'].map(std_meter_reading_per_building)





mean_meter_reading_per_meter = df_train.groupby('meter')['meter_reading'].mean()

df_train['mean_meter_reading_per_meter'] = df_train['meter'].map(mean_meter_reading_per_meter)

median_meter_reading_per_meter = df_train.groupby('meter')['meter_reading'].median()

df_train['median_meter_reading_per_meter'] = df_train['meter'].map(median_meter_reading_per_meter)

std_meter_reading_per_meter = df_train.groupby('meter')['meter_reading'].std()

df_train['std_meter_reading_per_meter'] = df_train['meter'].map(std_meter_reading_per_meter)





df_test['number_unique_meter_per_building'] = df_test['building_id'].map(number_unique_meter_per_building)



df_test['mean_meter_reading_per_building'] = df_test['building_id'].map(mean_meter_reading_per_building)

df_test['median_meter_reading_per_building'] = df_test['building_id'].map(median_meter_reading_per_building)

df_test['std_meter_reading_per_building'] = df_test['building_id'].map(std_meter_reading_per_building)



df_test['mean_meter_reading_per_meter'] = df_test['meter'].map(mean_meter_reading_per_meter)

df_test['median_meter_reading_per_meter'] = df_test['meter'].map(median_meter_reading_per_meter)

df_test['std_meter_reading_per_meter'] = df_test['meter'].map(std_meter_reading_per_meter)
%%time

for df in [df_train, df_test]:

    df['mean_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")

    df['median_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")

    df['std_meter_reading_per_building'] = df['std_meter_reading_per_building'].astype("float16")

    

    df['mean_meter_reading_per_meter'] = df['mean_meter_reading_per_meter'].astype("float16")

    df['median_meter_reading_per_meter'] = df['median_meter_reading_per_meter'].astype("float16")

    df['std_meter_reading_per_meter'] = df['std_meter_reading_per_meter'].astype("float16")

    

    df['number_unique_meter_per_building'] = df['number_unique_meter_per_building'].astype('uint8')

    df['square_feet'] = df['square_feet'].astype('float16')

gc.collect()
df_train.head()
df_train.drop('timestamp',axis=1,inplace=True)

df_test.drop('timestamp',axis=1,inplace=True)
print (df_train.shape, df_test.shape)
%%time

le = LabelEncoder()



df_train['meter']= le.fit_transform(df_train['meter']).astype("uint8")

df_test['meter']= le.fit_transform(df_test['meter']).astype("uint8")

df_train['primary_use']= le.fit_transform(df_train['primary_use']).astype("uint8")

df_test['primary_use']= le.fit_transform(df_test['primary_use']).astype("uint8")
y = df_train['meter_reading']

df_train.drop('meter_reading',axis=1,inplace=True)
categorical_cols = ['building_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth','site_id']
%%time

x_train,x_test,y_train,y_test = train_test_split(df_train,y,test_size=0.25,random_state=42)

print (x_train.shape)

print (y_train.shape)

print (x_test.shape)

print (y_test.shape)



lgb_train = lgb.Dataset(x_train, y_train,categorical_feature=categorical_cols)

lgb_test = lgb.Dataset(x_test, y_test,categorical_feature=categorical_cols)

del x_train, x_test , y_train, y_test



params = {'feature_fraction': 0.75,

          'bagging_fraction': 0.75,

          'objective': 'regression',

          'max_depth': -1,

          'learning_rate': 0.15,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'rmse',

          "verbosity": -1,

          'reg_alpha': 0.5,

          'reg_lambda': 0.5,

          'random_state': 47

         }



reg = lgb.train(params, lgb_train, num_boost_round=150, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval = 100)
del lgb_train,lgb_test
ser = pd.DataFrame(reg.feature_importance(),df_train.columns,columns=['Importance']).sort_values(by='Importance')

ser['Importance'].plot(kind='bar',figsize=(10,6))

del df_train
%%time

predictions = []

step = 150

for i in range(0, len(df_test), step):

    predictions.extend(np.expm1(reg.predict(df_test.iloc[i: min(i+step, len(df_test)), :], num_iteration=reg.best_iteration)))
Submission = pd.DataFrame(df_test.index,columns=['row_id'])
%%time

Submission['meter_reading'] = predictions

Submission['meter_reading'].clip(lower=0,upper=None,inplace=True)

Submission.to_csv("Twentysix.csv",index=None)