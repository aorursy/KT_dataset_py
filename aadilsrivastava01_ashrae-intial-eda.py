import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import gc
df_build = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

df_weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

df_train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')



df_weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

df_test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
df_build.head()
df_weather_train.head()
df_train.head()
print('Building Metadata', df_build.shape)

print('Train Data', df_train.shape)

print('Weather Train Data', df_weather_train.shape)



print('Test Data', df_test.shape)

print('Weather Test Data', df_weather_test.shape)
df_train.dtypes
df_weather_train.dtypes
def change_dtype(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    return df
df_train = change_dtype(df_train)

df_weather_train = change_dtype(df_weather_train)

df_test = change_dtype(df_test)

df_weather_test = change_dtype(df_weather_test)
df_train = df_train.merge(df_build, on='building_id', how='left')

df_train = df_train.merge(df_weather_train, on=['site_id', 'timestamp'], how='left')
df_test = df_test.merge(df_build, on='building_id', how='left')

df_test = df_test.merge(df_weather_test, on=['site_id', 'timestamp'], how='left')
del df_build, df_weather_test

gc.collect()
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
df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
df_weather_train = reduce_mem_usage(df_weather_train)
df_train.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

color = sns.color_palette()

sns.set_style("darkgrid")
df_train.shape
df_train.isna().sum()
len(df_train['building_id'].value_counts())
print(f"Max number of entries for is for building {df_train['building_id'].value_counts().keys()[0]}: ", df_train['building_id'].value_counts().values[0])

print(f"Min number of entries for is for building {df_train['building_id'].value_counts().keys()[-1]}: ", df_train['building_id'].value_counts().values[-1])
print("Data in the train set is from:", df_train['timestamp'].min())

print("Data in the train set is till:", df_train['timestamp'].max())
df_test['building_id'].value_counts()
print(f"Max number of entries for is for building {df_test['building_id'].value_counts().keys()[0]}: ", df_test['building_id'].value_counts().values[0])

print(f"Min number of entries for is for building {df_test['building_id'].value_counts().keys()[-1]}: ", df_test['building_id'].value_counts().values[-1])

print(f"Data in the train set is from:", df_test['timestamp'].min())

print(f"Data in the train set is till:", df_test['timestamp'].max())
df_train['meter'].value_counts().plot.bar(figsize=(8,4), color=color)

plt.show()
fig = plt.figure(figsize=(16, 12))



ax1 = fig.add_subplot(221)

sns.distplot(np.log1p(df_train[df_train['meter'] == 0]['meter_reading'].values),color='blue',ax=ax1, axlabel="Log of Electricity meter reading")

ax2 = fig.add_subplot(222)

sns.distplot(np.log1p(df_train[df_train['meter'] == 1]['meter_reading'].values),color='orange',ax=ax2, axlabel="Log of Chilledwater meter reading")

ax3 = fig.add_subplot(223)

sns.distplot(np.log1p(df_train[df_train['meter'] == 2]['meter_reading'].values),color='green',ax=ax3, axlabel="Log of Steam meter reading")

ax4 = fig.add_subplot(224)

sns.distplot(np.log1p(df_train[df_train['meter'] == 3]['meter_reading'].values),color='red',ax=ax4, axlabel="Log of Hotwater meter reading")



ax1.title.set_text('Meter Reading Distribution for Electricity Meter')

ax2.title.set_text('Meter Reading Distribution for Chilledwater Meter')

ax3.title.set_text('Meter Reading Distribution for Steam Meter')

ax4.title.set_text('Meter Reading Distribution for Hotwater Meter')



plt.show()
df_train.head()
df_temp = df_train.groupby(['building_id']).agg({"meter": ["nunique", "count"]})

df_temp.columns = ["_".join(i) for i in df_temp.columns.ravel()]
df_temp.sample(5)
df_temp['meter_nunique'].value_counts().plot.bar(figsize=(8,4), color=color)

plt.show()
df_temp[df_temp['meter_nunique']==4]
df_temp = df_train[df_train['building_id']==1258]

fig = plt.figure(figsize=(16, 16))



ax1 = fig.add_subplot(411)

df = df_temp[df_temp['meter']==0]

sns.lineplot(x=df['timestamp'], y=df['meter_reading'], ax=ax1, color='green')

ax2 = fig.add_subplot(412)

df = df_temp[df_temp['meter']==1]

sns.lineplot(x=df['timestamp'], y=df['meter_reading'], ax=ax2, color='green')

ax3 = fig.add_subplot(413)

df = df_temp[df_temp['meter']==2]

sns.lineplot(x=df['timestamp'], y=df['meter_reading'], ax=ax3, color='green')

ax4 = fig.add_subplot(414)

df = df_temp[df_temp['meter']==3]

sns.lineplot(x=df['timestamp'], y=df['meter_reading'], ax=ax4, color='green')



plt.tight_layout()

plt.subplots_adjust(top=1.5)

ax1.title.set_text('Meter Reading for Electricity Meter')

ax2.title.set_text('Meter Reading for Chilledwater Meter')

ax3.title.set_text('Meter Reading for Steam Meter')

ax4.title.set_text('Meter Reading for Hotwater Meter')



plt.show()
df_temp = df_train[df_train['building_id']==1331]

fig = plt.figure(figsize=(16, 16))



ax1 = fig.add_subplot(411)

df = df_temp[df_temp['meter']==0]

sns.lineplot(x=df['timestamp'], y=df['meter_reading'], ax=ax1)

ax2 = fig.add_subplot(412)

df = df_temp[df_temp['meter']==1]

sns.lineplot(x=df['timestamp'], y=df['meter_reading'], ax=ax2)

ax3 = fig.add_subplot(413)

df = df_temp[df_temp['meter']==2]

sns.lineplot(x=df['timestamp'], y=df['meter_reading'], ax=ax3)

ax4 = fig.add_subplot(414)

df = df_temp[df_temp['meter']==3]

sns.lineplot(x=df['timestamp'], y=df['meter_reading'], ax=ax4)



plt.tight_layout()

plt.subplots_adjust(top=1.5)

ax1.title.set_text('Meter Reading for Electricity Meter')

ax2.title.set_text('Meter Reading for Chilledwater Meter')

ax3.title.set_text('Meter Reading for Steam Meter')

ax4.title.set_text('Meter Reading for Hotwater Meter')



plt.show()
fig = plt.figure(figsize=(22, 8))

df_train['year_built'].value_counts().sort_index().plot.bar(color=color)

plt.show()
fig = plt.figure(figsize=(16, 8))

df_train['primary_use'].value_counts(dropna=False).plot.bar(color=color)

plt.show()
fig = plt.figure(figsize=(16, 8))

df_train['floor_count'].value_counts(normalize=True, dropna=False).plot.bar(color=color)

plt.show()
print("Smallest building Area: ", df_train['square_feet'].min())

print("largetst building Area: ", df_train['square_feet'].max())

print("Mean building Area: ", df_train['square_feet'].mean())

print("Median building Area: ", df_train['square_feet'].median())
fig = plt.figure(figsize=(16, 8))

sns.scatterplot(x=df_train['year_built'], y=df_train['square_feet'])

plt.show()
df_weather_train.shape
df_weather_train.isnull().sum()
fig = plt.figure(figsize=(8,6))

sns.distplot(df_weather_train['air_temperature'].dropna(), color='red', axlabel="Air Temprature")

plt.show()
fig = plt.figure(figsize=(8,6))

sns.distplot(df_weather_train['dew_temperature'].dropna(), color='green', axlabel="Dew Temprature")

plt.show()
fig = plt.figure(figsize=(8,6))

sns.distplot(df_weather_train['sea_level_pressure'].dropna(), color='blue', axlabel="Sea Level Pressure")

plt.show()
fig = plt.figure(figsize=(8,6))

sns.distplot(df_weather_train['wind_speed'].dropna(), color='purple', axlabel="Wind Speed")

plt.show()
fig = plt.figure(figsize=(16,8))

df_train['site_id'].value_counts(normalize=True, dropna=False).plot.bar(color=color)

plt.show()
fig = plt.figure(figsize=(22,11))

sns.lineplot(x=df_weather_train['timestamp'], y=df_weather_train['air_temperature'], hue=df_weather_train['site_id'], 

             legend='full', palette=sns.color_palette('Paired',16))

plt.show()
fig = plt.figure(figsize=(22,11))

sns.lineplot(x=df_weather_train['timestamp'], y=df_weather_train['dew_temperature'], hue=df_weather_train['site_id'], 

             legend='full', palette=sns.color_palette('Paired',16))

plt.show()
fig = plt.figure(figsize=(22,11))

sns.lineplot(x=df_weather_train['timestamp'], y=df_weather_train['sea_level_pressure'], hue=df_weather_train['site_id'], 

             legend='full', palette=sns.color_palette('Paired',16))

plt.show()
fig = plt.figure(figsize=(22,11))

sns.lineplot(x=df_weather_train['timestamp'], y=df_weather_train['wind_speed'], hue=df_weather_train['site_id'], 

             legend='full', palette=sns.color_palette('Paired',16))

plt.show()