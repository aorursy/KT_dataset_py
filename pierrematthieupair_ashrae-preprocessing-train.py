import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

import pickle

from sklearn.preprocessing import LabelEncoder

import sys

import datetime
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
## Function to reduce the DF size

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
# Original imputing code for weather from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling by @aitude

def fill_weather_dataset(weather_df):

    

    # Find Missing Dates

    time_format = "%Y-%m-%d %H:%M:%S"

    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)

    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]



    missing_hours = []

    for site_id in range(16):

        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])

        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])

        new_rows['site_id'] = site_id

        weather_df = pd.concat([weather_df,new_rows])



        weather_df = weather_df.reset_index(drop=True)           



    # Add new Features

    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])

    weather_df["day"] = weather_df["datetime"].dt.day

    weather_df["week"] = weather_df["datetime"].dt.week

    weather_df["month"] = weather_df["datetime"].dt.month

    

    # Reset Index for Fast Update

    weather_df = weather_df.set_index(['site_id','day','month'])



    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

    weather_df.update(air_temperature_filler,overwrite=False)



    # Step 1

    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()

    # Step 2

    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])



    weather_df.update(cloud_coverage_filler,overwrite=False)



    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

    weather_df.update(due_temperature_filler,overwrite=False)



    # Step 1

    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()

    # Step 2

    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])



    weather_df.update(sea_level_filler,overwrite=False)



    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

    weather_df.update(wind_direction_filler,overwrite=False)



    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])

    weather_df.update(wind_speed_filler,overwrite=False)



    # Step 1

    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()

    # Step 2

    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])



    weather_df.update(precip_depth_filler,overwrite=False)



    weather_df = weather_df.reset_index()

    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)

        

    return weather_df
building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

weather = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')       
def basic_preprocessing(building_metadata, weather, data):



    data = reduce_mem_usage(data)

    building_metadata = reduce_mem_usage(building_metadata)

    weather = reduce_mem_usage(weather)



    # joining by building_id

    data = (building_metadata.set_index("building_id").join(data.set_index("building_id"))).reset_index()



    # Correct units for site 0 to kwh    

    data.loc[(data['site_id'] == 0) & (data['meter'] == 0), 'meter_reading'] = data[(data['site_id'] == 0) & (data['meter'] == 0)]['meter_reading'] * 0.2931    

    

    # joining by site_id and timestamp using multi indexes

    data = data.set_index(['site_id','timestamp']).join(weather.set_index(['site_id','timestamp'])).reset_index()

    del building_metadata, weather

    gc.collect()

    

    # Convert timestamp string to datetime

    data.loc[:, 'timestamp'] = pd.to_datetime(data.timestamp)



    # Remove all rows where the meter reading is 0

    data = data.drop(data.loc[data.meter_reading == 0].index, axis = 0)

    

    data = reduce_mem_usage(data)   

    print(data.memory_usage().sum() / 1024**2, 'Mb')    

        

    return data
def preprocessing():

    building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

    data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

    weather = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')           

   

    weather = fill_weather_dataset(weather)



    data = reduce_mem_usage(data)

    building_metadata = reduce_mem_usage(building_metadata)

    weather = reduce_mem_usage(weather)



    # joining by building_id

    data = (building_metadata.set_index("building_id").join(data.set_index("building_id"))).reset_index()



    # Correct units for site 0 to kwh    

    data.loc[(data['site_id'] == 0) & (data['meter'] == 0), 'meter_reading'] = data[(data['site_id'] == 0) & (data['meter'] == 0)]['meter_reading'] * 0.2931    

    

    # joining by site_id and timestamp using multi indexes

    data = data.set_index(['site_id','timestamp']).join(weather.set_index(['site_id','timestamp'])).reset_index()

    del building_metadata, weather

    gc.collect()

    

    # Convert timestamp string to datetime

    data.loc[:, 'timestamp'] = pd.to_datetime(data.timestamp)

    data['month'] = pd.DatetimeIndex(data.timestamp).month

    data['weekday'] = pd.DatetimeIndex(data.timestamp).dayofweek

    data['hour'] = pd.DatetimeIndex(data.timestamp).hour

    data['day'] = pd.DatetimeIndex(data.timestamp).day



    # Remove outliers

    Meter1_Outliers = data.loc[(data.meter == 1) & (data.meter_reading > 20000)].building_id.unique()

    data = data[~data['building_id'].isin(Meter1_Outliers)] 

    Meter2_Outliers = data.loc[(data.meter == 2) & (data.meter_reading > 20000)].building_id.unique()

    data = data[~data['building_id'].isin(Meter2_Outliers)] 

    Meter3_Outliers = data.loc[(data.meter == 3) & (data.meter_reading > 5000)].building_id.unique()

    data = data[~data['building_id'].isin(Meter3_Outliers)] 



    # Remove all rows where the meter reading is 0

    data = data.drop(data.loc[data.meter_reading == 0].index, axis = 0)

    y = data.meter_reading

    data = data.drop('meter_reading', axis = 1)           

    

    # Dropping useless

    useless = ['timestamp', "sea_level_pressure", "wind_direction", "wind_speed","year_built","floor_count"]

    data = data.drop(useless, axis = 1)

    gc.collect()

    

    le = LabelEncoder()

    data["primary_use"] = le.fit_transform(data["primary_use"])

    output = open('LabelEncoder.pkl', 'wb')

    pickle.dump(le, output)

    output.close()

    

    output = open('data_train.pkl', 'wb')  

    pickle.dump(data, output)

    output.close()

    

    output = open('y.pkl', 'wb')  

    pickle.dump(y, output)

    output.close()

    

    data = reduce_mem_usage(data)   

    print(data.memory_usage().sum() / 1024**2, 'Mb')    

        

    return data, y
data = basic_preprocessing(building_metadata, weather, data)
# Prepares and saves preprocessed data for use by the model notebook

preprocessing()
# Global data. Pretty obvious outliers in the meter_readings. 

#But where is the threshold between large building consumption and a wrong measurement ?

data.describe()
# Data for power consumption (meter = 0)

data.loc[(data.meter == 0)].describe()
plt.figure(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.boxplot(x=data.loc[(data.meter == 0)].meter_reading)
# Limiting ourselves to reasonable values

plt.figure(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.boxplot(x=data.loc[(data.meter == 0) & (data.meter_reading < 1000)].meter_reading)
# How many buildings have large values? which ones?

data.loc[(data.meter == 0) & (data.meter_reading > 400)].nunique()
data.loc[(data.meter == 0) & (data.meter_reading > 400)].building_id.unique()
data.loc[(data.meter == 0) & (data.meter_reading > 400)].describe()
# Data for chilled water consumption (meter = 1)

data.loc[(data.meter == 1)].describe()
plt.figure(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.boxplot(x=data.loc[(data.meter == 1)].meter_reading)
plt.figure(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.boxplot(x=data.loc[(data.meter == 1) & (data.meter_reading < 3000)].meter_reading)
data.loc[(data.meter == 1) & (data.meter_reading > 20000)].nunique()
data.loc[(data.meter == 1) & (data.meter_reading > 20000)].building_id.unique()
PotentialOutliers = data.loc[(data.meter == 1) & (data.meter_reading > 20000)]

PotentialOutliers.groupby(['building_id']).apply(lambda df: df.loc[df.meter_reading.idxmax()])
ax = sns.scatterplot(x=PotentialOutliers.building_id, y = PotentialOutliers.meter_reading )
# Data for steam consumption (meter = 2)

data.loc[(data.meter == 2)].describe()
plt.figure(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.boxplot(x=data.loc[(data.meter == 2)].meter_reading)
plt.figure(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.boxplot(x=data.loc[(data.meter == 2) & (data.meter_reading < 5000)].meter_reading)
data.loc[(data.meter == 2) & (data.meter_reading > 20000)].nunique()
data.loc[(data.meter == 2) & (data.meter_reading > 20000)].building_id.unique()
PotentialOutliers = data.loc[(data.meter == 2) & (data.meter_reading > 20000)]

PotentialOutliers.groupby(['building_id']).apply(lambda df: df.loc[df.meter_reading.idxmax()])
ax = sns.scatterplot(x=PotentialOutliers.building_id, y = PotentialOutliers.meter_reading )
# Data for hot water consumption (meter = 3)

data.loc[(data.meter == 3)].describe()
plt.figure(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.boxplot(x=data.loc[(data.meter == 3)].meter_reading)
plt.figure(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.boxplot(x=data.loc[(data.meter == 3) & (data.meter_reading < 5000)].meter_reading)
data.loc[(data.meter == 3) & (data.meter_reading > 5000)].nunique()
data.loc[(data.meter == 3) & (data.meter_reading > 5000)].building_id.unique()
PotentialOutliers = data.loc[(data.meter == 3) & (data.meter_reading > 5000)]

PotentialOutliers.groupby(['building_id']).apply(lambda df: df.loc[df.meter_reading.idxmax()])

ax = sns.scatterplot(x=PotentialOutliers.building_id, y = PotentialOutliers.meter_reading )
plt.figure(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.boxplot(x=building_metadata.square_feet)
# Concentrating on big buildings (for identifying the outliers)

Big = building_metadata.loc[building_metadata.square_feet > 600000]

Big.nunique()
weather.isna().sum()
weather.loc[:, 'timestamp'] = pd.to_datetime(weather.timestamp)

weather.loc[:, 'timestamp'] = pd.to_datetime(weather.timestamp)

weather['month'] = pd.DatetimeIndex(weather.timestamp).month

weather['weekday'] = pd.DatetimeIndex(weather.timestamp).dayofweek

weather['hour'] = pd.DatetimeIndex(weather.timestamp).hour

weather['day'] = pd.DatetimeIndex(weather.timestamp).day

weather['date'] = pd.to_timedelta(weather.timestamp).dt.total_seconds() / 3600

weather['date'] = weather.date.astype(int)

weather.date -= weather.date.min()

weather.date.describe()
# Air temp

plt.figure(figsize=(14,7))

missmap = np.empty((16, weather.date.max()+1))

missmap.fill(np.nan)

for l in weather.values:

    missmap[int(l[0]), int(l[13])] = l[2]

sns.heatmap(missmap)
# cloud : site 7 and 11 have no values

plt.figure(figsize=(14,7))

missmap = np.empty((16, weather.date.max()+1))

missmap.fill(np.nan)

for l in weather.values:

    missmap[int(l[0]), int(l[13])] = l[3]

sns.heatmap(missmap)
# precipitation : sites 1, 5, 12 have none

plt.figure(figsize=(14,7))

missmap = np.empty((16, weather.date.max()+1))

missmap.fill(np.nan)

for l in weather.values:

    missmap[int(l[0]), int(l[13])] = l[5]

sns.heatmap(missmap)
# pressure : site 5 missing

plt.figure(figsize=(14,7))

missmap = np.empty((16, weather.date.max()+1))

missmap.fill(np.nan)

for l in weather.values:

    missmap[int(l[0]), int(l[13])] = l[6]

sns.heatmap(missmap)
# wind dir

plt.figure(figsize=(14,7))

missmap = np.empty((16, weather.date.max()+1))

missmap.fill(np.nan)

for l in weather.values:

    missmap[int(l[0]), int(l[13])] = l[7]

sns.heatmap(missmap)