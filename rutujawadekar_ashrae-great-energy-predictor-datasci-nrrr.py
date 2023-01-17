# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

sample_submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")
train.info()

test.info()

weather_train.info()

weather_test.info()

building_metadata.info()
######### Change the Data types of objects into more specific data types ###############



train['timestamp'] = pd.to_datetime(train['timestamp'])



test['timestamp'] = pd.to_datetime(test['timestamp'])



weather_train['timestamp'] = pd.to_datetime (weather_train['timestamp'])



weather_test ['timestamp']= pd.to_datetime(weather_test['timestamp'])



building_metadata ['primary_use'] = building_metadata['primary_use'].astype('category')
################################################################################

######### Reducing the memory size of the train and test dataset ##############

################################################################################



def reduce_mem_usage(data):

    types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem_usg = data.memory_usage ().sum () / 1024 ** 2

    print ("Memory usage of properties dataframe is :", start_mem_usg, " MB")



    for col in data.columns:

        if data[col].dtype in types:  # Exclude strings

            mx = data[col].max()

            mn = data[col].min()



            # Make Integer/unsigned Integer datatypes

            if str(data[col].dtype)[:3] == 'int':

                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                    data[col] = data[col].astype(np.int8)

                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                    data[col] = data[col].astype(np.int16)

                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                    data[col] = data[col].astype(np.int32)

                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                    data[col] = data[col].astype(np.int64)

            else:

                if mn > np.finfo(np.float16).min and mx < np.finfo(np.float16).max:

                    data[col] = data[col].astype(np.float16)

                elif mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max:

                    data[col] = data[col].astype(np.float32)

                elif mn > np.finfo(np.float64).min and mx < np.finfo(np.float64).max:

                    data[col] = data[col].astype(np.float64)



        # Print final result

    print ("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = data.memory_usage ().sum () / 1024 ** 2

    print ("Memory usage is: ", mem_usg, " MB")

    print ("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")

    return data
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

weather_train = reduce_mem_usage(weather_train)

weather_test = reduce_mem_usage(weather_test)

building_metadata = reduce_mem_usage(building_metadata)
train_missing = (train.isnull().sum()/len(train))*100

test_missing = (test.isnull().sum()/len(test))*100

weatherTrain_missing = (weather_train.isnull().sum()/len(weather_train))*100

weatherTest_missing = (weather_test.isnull().sum()/len(weather_test))*100

metadata_missing = (building_metadata.isnull().sum()/ len(building_metadata))*100
test.head()

missing_count = pd.concat([train_missing, test_missing],keys = ['missing train','missing test'], axis = 1, sort = False)

print(missing_count)
print('missing % of building metadata\n',metadata_missing)
missing_weather_count = pd.concat([weatherTrain_missing, weatherTest_missing], keys= ['missing % weather train', 'missing % weather test'], axis =1 , sort = False)

print(missing_weather_count)
weather_train.head()
weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

weather_train["datetime"] = pd.to_datetime(weather_train["timestamp"])

weather_train["day"] = weather_train["datetime"].dt.day

weather_train["week"] = weather_train["datetime"].dt.week

weather_train["month"] = weather_train["datetime"].dt.month



weather_test["datetime"] = pd.to_datetime(weather_test["timestamp"])

weather_test["day"] = weather_test["datetime"].dt.day

weather_test["week"] = weather_test["datetime"].dt.week

weather_test["month"] = weather_test["datetime"].dt.month





weather_train = weather_train.set_index(['site_id','day','month'])

weather_test = weather_test.set_index(['site_id','day','month'])



air_temperature_filler = pd.DataFrame(weather_train.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

weather_train.update(air_temperature_filler,overwrite=False)



air_temperature_filler_test = pd.DataFrame(weather_test.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

weather_test.update(air_temperature_filler_test)



cloud_coverage_filler = weather_train.groupby(['site_id','day','month'])['cloud_coverage'].mean()

cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

weather_train.update(cloud_coverage_filler,overwrite=False)





cloud_coverage_filler_test = weather_test.groupby(['site_id','day','month'])['cloud_coverage'].mean()

cloud_coverage_filler_test = pd.DataFrame(cloud_coverage_filler_test.fillna(method='ffill'),columns=["cloud_coverage"])

weather_test.update(cloud_coverage_filler_test)





dew_temperature_filler = pd.DataFrame(weather_train.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

weather_train.update(dew_temperature_filler,overwrite=False)



dew_temperature_filler_test = pd.DataFrame(weather_test.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

weather_test.update(dew_temperature_filler_test)



wind_direction_filler =  pd.DataFrame(weather_train.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

weather_train.update(wind_direction_filler,overwrite=False)



wind_direction_filler_test =  pd.DataFrame(weather_test.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

weather_test.update(wind_direction_filler_test)





weather_train = weather_train.reset_index()

weather_train = weather_train.drop(['datetime','day','week','month'],axis=1)



weather_test = weather_test.reset_index()

weather_test = weather_test.drop(['datetime','day','week','month'],axis=1)

weatherTrain_missing = (weather_train.isnull().sum()/len(weather_train))*100

weatherTest_missing = (weather_test.isnull().sum()/len(weather_test))*100

missing_weather_count = pd.concat([weatherTrain_missing, weatherTest_missing], keys= ['missing % weather train', 'missing % weather test'], axis =1 , sort = False)

print(missing_weather_count)
building_metadata['floor_count'] =building_metadata['floor_count'].fillna(building_metadata['floor_count'].mean())

building_metadata['year_built'] = building_metadata['year_built'].fillna(building_metadata['year_built'].mean())
metadata_missing = (building_metadata.isnull().sum()/ len(building_metadata))*100

print(metadata_missing)
weather_train['timestamp'] = pd.to_datetime (weather_train['timestamp'])

weather_test ['timestamp']= pd.to_datetime(weather_test['timestamp'])
train = train.merge(building_metadata, on=['building_id'], how='left')

train.head()
test = test.merge(building_metadata, on=['building_id'], how='left')

test.head()
#train['timestamp'] = pd.to_datetime(train['timestamp'])

print(train.info())

print(weather_train.info())
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')

train.head()
test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')

test.head()
import matplotlib.pyplot as plt

fig = train.hist(figsize=(15, 10),)

[x.title.set_size(10) for x in fig.ravel()]

plt.tight_layout()

plt.show()
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import seaborn as sns

fig = plt.figure(figsize=(12,8))

train['meter'] = LabelEncoder().fit_transform(train['meter'])



# Separate both dataframes into 

numeric_df = train.select_dtypes(exclude="object")

# categorical_df = df.select_dtypes(include="object")



corr_numeric = numeric_df.corr()

sns.heatmap(corr_numeric, cbar=True, cmap="RdBu_r")

plt.title("Correlation Matrix", fontsize=16)

plt.show()
train[train.columns[1:]].corr()['meter'][:].sort_values(ascending=False)
plt.figure(figsize=(10,5))

plt.tight_layout()

sns.distplot(train['meter'])
fig, ax = plt.subplots(1, 2, figsize=(18,4))



square_feet = train['square_feet'].values

site_id = train['site_id'].values



sns.distplot(square_feet, ax=ax[0], color='r')

ax[0].set_title('Distribution of square_feet', fontsize=14)

ax[0].set_xlim([min(square_feet), max(square_feet)])



sns.distplot(site_id, ax=ax[1], color='b')

ax[1].set_title('Distribution of site_id', fontsize=14)

ax[1].set_xlim([min(site_id), max(site_id)])



plt.show()
train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
sns.countplot(train['meter'])

plt.title("Distribution of Meter Id Code")

plt.xlabel("Meter Id Code")

plt.ylabel("Frequency")
fig, axes = plt.subplots(1, 1, figsize=(18, 4))

sns.boxplot(x='meter', y='meter_reading', data=train, showfliers=False, palette="Set3");
fig, axes = plt.subplots(1, 1, figsize=(18, 4))

sns.boxplot(x='meter', y='air_temperature', data=train,  showfliers=False, palette="Set3");
tmp_df = pd.concat([train[['meter', 'site_id']]], ignore_index=True)

tmp_df['dataset'] = 'Train'



fig, axes = plt.subplots(1, 1, figsize=(10, 4))

sns.boxplot(x='meter', y='site_id', data=tmp_df, hue='dataset', palette="Set3");



del tmp_df
tmp_df = pd.concat([train[['meter', 'air_temperature']]], ignore_index=True)

tmp_df['dataset'] = 'Train'



fig, axes = plt.subplots(1, 1, figsize=(10, 4))

sns.boxplot(x='meter', y='air_temperature', data=tmp_df, hue='dataset', palette="Set3");



del tmp_df
sns.set(rc={'figure.figsize':(18, 4)})

train['meter_reading'].plot();
from statsmodels.tsa.seasonal import seasonal_decompose

ts=train.groupby(["timestamp"])["meter_reading"].mean().astype('float')

plt.figure(figsize=(18,4))

plt.title('Mean_reading vs RollingMean_reading')

plt.xlabel('Year_Month')

plt.ylabel('meter_reading')

plt.plot(ts,label='Mean');

plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.legend();
train['timestamp']= pd.to_datetime(train.timestamp) 

train = train.set_index('timestamp')
fig, axes = plt.subplots(1, 1, figsize=(18, 4), dpi=100)

train[['meter_reading']].resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train[['meter_reading']].resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('Mean Meter reading by hour and day');

axes.legend();
for i in range(train['site_id'].nunique()):

    ts1=train[train['site_id'] == i][['meter_reading']].resample('H').mean().astype('float')

    ts2=train[train['site_id'] == i][['meter_reading']].resample('D').mean().astype('float')

    plt.figure(figsize=(14,3))

    plt.title("Site id")

    plt.title(i)

    plt.xlabel('Year_Month')

    plt.ylabel('meter_reading')

    plt.plot(ts1,label='Hourly_Reading');

    plt.plot(ts2,label='Weekly_Reading');

    plt.legend(); 
plt.scatter(train['meter_reading'], train['air_temperature'], color='blue')

plt.title('Meter Reading Vs Air Temperature', fontsize=14)

plt.xlabel('Meter Reading', fontsize=14)

plt.ylabel('Air Temperature', fontsize=14)

plt.grid(True)

plt.show()
plt.scatter(train['meter_reading'], train['cloud_coverage'], color='blue')

plt.title('Meter Reading Vs Cloud Coverage', fontsize=14)

plt.xlabel('Meter Reading', fontsize=14)

plt.ylabel('Cloud Coverage', fontsize=14)

plt.grid(True)

plt.show()
plt.scatter(train['meter_reading'], train['dew_temperature'], color='red')

plt.title('Meter Reading Vs Dew Temperature', fontsize=14)

plt.xlabel('Meter Reading', fontsize=14)

plt.ylabel('Dew Temperature', fontsize=14)

plt.grid(True)

plt.show()