from collections import defaultdict

from datetime import datetime, timedelta

from tqdm import tqdm

import holidays

import gc

import os

import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks as cf

cf.go_offline(connected=False)



import lightgbm as lgb

print(lgb.__version__)



SEED = 42
WEATHER_DTYPES = {'site_id': np.uint8, 'air_temperature': np.float32, 'cloud_coverage': np.float32, 'dew_temperature': np.float32, 

                     'precip_depth_1_hr': np.float32, 'sea_level_pressure': np.float32, 'wind_direction': np.float32, 'wind_speed': np.float32}

df_weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv', dtype=WEATHER_DTYPES)

df_weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv', dtype=WEATHER_DTYPES)



df_weather = pd.concat([df_weather_train, df_weather_test], ignore_index=True)

df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'], infer_datetime_format=True)



print('Weather Set Shape = {}'.format(df_weather.shape))

print('Weather Set Memory Usage = {:.2f} MB'.format(df_weather.memory_usage().sum() / 1024**2))

print('Training Weather Set Time Period = {} - {}'.format(df_weather[:len(df_weather_train)]['timestamp'].min(), df_weather[:len(df_weather_train)]['timestamp'].max()))

print('Test Weather Set Time Period = {} - {}'.format(df_weather[len(df_weather_train):]['timestamp'].min(), df_weather[len(df_weather_train):]['timestamp'].max()))
df_weather['HourGap'] = df_weather.groupby('site_id')['timestamp'].diff() / np.timedelta64(1, 'h')



plt.figure(figsize=(25, 15))

for i in df_weather['site_id'].unique():

    ax = plt.subplot(4, 4, i + 1)    

    df_weather[df_weather['site_id'] == i].set_index('timestamp')['HourGap'].plot()

    ax.set_title(f'site {i} Hour Gaps')

    

plt.tight_layout()

plt.show()



df_weather.drop(columns=['HourGap'], inplace=True)
weather_key = ['site_id', 'timestamp']

df_air_temperature = df_weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()

df_air_temperature['HourOfDay'] = df_air_temperature['timestamp'].dt.hour



plt.figure(figsize=(25, 15))

for site_id, data_by_site in df_air_temperature.groupby('site_id'):

    mean = data_by_site.groupby('HourOfDay').mean()

    

    ax = plt.subplot(4, 4, site_id + 1)

    plt.plot(mean.index, mean['air_temperature'], 'xb-')

    ax.set_title(f'site {site_id} Hour Mean Temperature')

    

plt.tight_layout()

plt.show()
df_air_temperature['air_temperature_rank'] = df_air_temperature.groupby(['site_id', df_air_temperature['timestamp'].dt.date])['air_temperature'].rank('average')

df_air_temperature_rank = df_air_temperature.groupby(['site_id', 'HourOfDay'])['air_temperature_rank'].mean().unstack(level=1)

df_air_temperature_rank = df_air_temperature_rank / df_air_temperature_rank.max(axis=1).values.reshape((-1,1))



site_ids_argmax_maxtemp = pd.Series(np.argmax(df_air_temperature_rank.values, axis=1)).sort_values().index

site_ids_offsets = pd.Series(df_air_temperature_rank.values.argmax(axis=1) - 14)



df_air_temperature_rank = df_air_temperature_rank.iloc[site_ids_argmax_maxtemp]

df_air_temperature_rank.index = [f'idx={i:02d}_site_id={s:02d}' for (i, s) in zip(range(16), df_air_temperature_rank.index)]



df_air_temperature_rank.T.iplot(kind='heatmap', colorscale='ylorrd', xTitle='HourOfDay [0-23]', title='Mean air_temperature Rank by Hour from Least to Most Correct Site')
df_air_temperature['offset'] = df_air_temperature['site_id'].map(site_ids_offsets)

df_air_temperature['timestamp_aligned'] = (df_air_temperature['timestamp'] - pd.to_timedelta(df_air_temperature['offset'], unit='H'))

df_air_temperature['air_temperature_rank_aligned'] = df_air_temperature.groupby(['site_id', df_air_temperature['timestamp_aligned'].dt.date])['air_temperature'].rank('max')



# Adding the timestamp_aligned to the df_weather

df_weather['timestamp'] = df_air_temperature['timestamp_aligned']



df_air_temperature_rank = df_air_temperature.groupby(['site_id', df_air_temperature['timestamp_aligned'].dt.hour])['air_temperature_rank_aligned'].mean().unstack(level=1)

df_air_temperature_rank.T.iplot(kind='heatmap', colorscale='ylorrd', xTitle='HourOfDay [0-23]', yTitle='site_id [0-16]', title='Mean air_temperature Rank by Hour with Aligned Timestamps')



del df_air_temperature, df_air_temperature_rank, site_ids_argmax_maxtemp, site_ids_offsets
print('Initial Weather Set Shape = {}'.format(df_weather.shape))



# Setting site_id and timestamp as multi index and creating missing hours

site_ids = sorted(np.unique(df_weather['site_id']))

df_weather = df_weather.set_index(['site_id', 'timestamp'], drop=False).sort_index()

full_index = pd.MultiIndex.from_product([site_ids, pd.date_range(start='2015-12-31 15:00:00', end='2018-12-31 23:00:00', freq='H')])

df_weather = df_weather.reindex(full_index)

print('Weather Set Shape after reindexing = {}'.format(df_weather.shape))



# timestamp and site_id as features again

df_weather['site_id'] = df_weather.index.get_level_values(0)

df_weather['site_id'] = df_weather['site_id'].astype(np.uint8)

df_weather['timestamp'] = df_weather.index.get_level_values(1)



# Categorical date and time features

df_weather['HourOfDay'] = df_weather['timestamp'].dt.hour.values.astype(np.uint8)

df_weather['DayOfWeek'] = df_weather['timestamp'].dt.dayofweek.values.astype(np.uint8)

df_weather['DayOfMonth'] = df_weather['timestamp'].dt.day.values.astype(np.uint8)

df_weather['DayOfYear'] = df_weather['timestamp'].dt.dayofyear.values.astype(np.uint16)

df_weather['WeekOfYear'] = (np.floor(df_weather['DayOfYear'] / 7) + 1).astype(np.uint8) # Series.dt.weekofyear is not correct: https://github.com/pandas-dev/pandas/issues/6936

df_weather['MonthOfYear'] = df_weather['timestamp'].dt.month.values.astype(np.uint8)

df_weather['Year'] = df_weather['timestamp'].dt.year.astype(np.uint16)



# Continuous date and time features

df_weather['Hour'] = ((pd.to_timedelta(df_weather['timestamp'] - df_weather['timestamp'].min()).dt.total_seconds().astype('int64')) / 3600).astype(np.uint16)

df_weather['Day'] = (df_weather['Hour'] / 24).astype(np.uint16)

df_weather['Week'] = (df_weather['Day'] / 7).astype(np.uint8)



print('Weather Set Shape after date and time features = {}'.format(df_weather.shape))
WEATHER_COLS = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']



fig = plt.figure(figsize=(15, 7))

sns.barplot(x=df_weather[WEATHER_COLS].isnull().sum().index, y=df_weather[WEATHER_COLS].isnull().sum().values)



plt.xlabel('Weather Features', size=15, labelpad=20)

plt.ylabel('Missing Value Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=15)



plt.show()
def nan_imputer(col, window=24):

    

    WEATHER_PREDICTORS = ['HourOfDay', 'DayOfYear', 'WeekOfYear', 'MonthOfYear',  'Year', 'site_id', 

                          f'Site_{col}_RollingBackMean', f'Site_{col}_RollingForwMean', f'SiteHourOfDay_{col}_RollingBackMean', f'SiteHourOfDay_{col}_RollingForwMean']

    df = df_weather.copy()

    

    imputer = lgb.LGBMRegressor(

        learning_rate=0.05,

        objective='mae',

        n_estimators=350,

        num_threads=os.cpu_count(),

        num_leaves=31,

        max_depth=8,

        subsample=0.8,

        min_child_samples=50,

        random_state=SEED,

    )    

    

    df[f'Site_{col}_RollingBackMean'] = df.groupby('site_id')[col].rolling(window=window, min_periods=1).mean().interpolate().values

    df[f'Site_{col}_RollingForwMean'] = df.iloc[::-1].groupby('site_id')[col].rolling(window=window, min_periods=1).mean().interpolate().values

    df[f'SiteHourOfDay_{col}_RollingBackMean'] = df.groupby(by=['site_id', 'HourOfDay'])[col].rolling(window=3, min_periods=1).mean().interpolate().values

    df[f'SiteHourOfDay_{col}_RollingForwMean'] = df.iloc[::-1].groupby(by=['site_id', 'HourOfDay'])[col].rolling(window=3, min_periods=1).mean().interpolate().values

    

    trn_idx, missing_idx = ~df[col].isnull(), df[col].isnull()    

    imputer.fit(X=df.loc[trn_idx, WEATHER_PREDICTORS], y=df.loc[trn_idx, col], categorical_feature=['site_id', 'Year'])

    

    df[f'{col}_Restored'] = df[col].copy()

    df.loc[missing_idx, f'{col}_Restored'] = imputer.predict(df.loc[missing_idx, WEATHER_PREDICTORS])

    

    lgb.plot_importance(imputer)

    plt.title(f'{col} Imputation Feature Importance')

    

    return df[f'{col}_Restored'].values.astype(np.float32)



# Makes imputation on a copy and visualizes it

def plot_imputation(col, start='2016-01-01 00:00:00', end='2018-12-31 23:00:00'):    

        

    df = df_weather.copy()

    df[f'{col}_Restored'] = nan_imputer(col, window=24)

    df[f'{col}_RollingMean'] = df.groupby(by='site_id')[col].rolling(window=(24 * 3), min_periods=1).mean().values



    for site in range(16):

        start =  (site, '2016-01-01 00:00:00')

        end = (site, '2018-07-01 00:00:00')

        df.loc[start:end].set_index('timestamp')[[f'{col}_Restored', f'{col}_RollingMean', f'{col}']].iplot()    

    
df_weather['air_temperature'] = nan_imputer('air_temperature')

df_weather['dew_temperature'] = nan_imputer('dew_temperature')

df_weather['wind_speed'] = nan_imputer('wind_speed')



df_weather.drop(columns=['DayOfMonth', 'DayOfYear', 'WeekOfYear', 'MonthOfYear', 'Year', 'Hour', 'Day', 'Week'], inplace=True)
# Humidity

saturated_vapor_pressure = 6.11 * (10.0 ** (7.5 * df_weather['air_temperature'] / (237.3 + df_weather['air_temperature'])))                                    

actual_vapor_pressure = 6.11 * (10.0 ** (7.5 * df_weather['dew_temperature'] / (237.3 + df_weather['dew_temperature'])))    

df_weather['humidity'] = (actual_vapor_pressure / saturated_vapor_pressure) * 100

df_weather['humidity'] = df_weather['humidity'].astype(np.float32)



del saturated_vapor_pressure, actual_vapor_pressure

gc.collect()



# Rolling Weather Features

#for col in ['air_temperature', 'dew_temperature', 'humidity']:

    #df_weather[col + '_mean_1'] = df_weather.groupby('site_id')[col].rolling(24).mean().values

    #df_weather[col + '_mean_3'] = df_weather.groupby('site_id')[col].rolling(72).mean().values

    #df_weather[col + '_std_1'] = df_weather.groupby('site_id')[col].rolling(24).std().values

    #df_weather[col + '_std_3'] = df_weather.groupby('site_id')[col].rolling(72).std().values
en_holidays = holidays.England()

ir_holidays = holidays.Ireland()

ca_holidays = holidays.Canada()

us_holidays = holidays.UnitedStates()



en_idx = df_weather.query('site_id == 1 or site_id == 5').index

ir_idx = df_weather.query('site_id == 12').index

ca_idx = df_weather.query('site_id == 7 or site_id == 11').index

us_idx = df_weather.query('site_id == 0 or site_id == 2 or site_id == 3 or site_id == 4 or site_id == 6 or site_id == 8 or site_id == 9 or site_id == 10 or site_id == 13 or site_id == 14 or site_id == 15').index



df_weather['IsHoliday'] = 0

df_weather.loc[en_idx, 'IsHoliday'] = df_weather.loc[en_idx, 'timestamp'].apply(lambda x: en_holidays.get(x, default=0))

df_weather.loc[ir_idx, 'IsHoliday'] = df_weather.loc[ir_idx, 'timestamp'].apply(lambda x: ir_holidays.get(x, default=0))

df_weather.loc[ca_idx, 'IsHoliday'] = df_weather.loc[ca_idx, 'timestamp'].apply(lambda x: ca_holidays.get(x, default=0))

df_weather.loc[us_idx, 'IsHoliday'] = df_weather.loc[us_idx, 'timestamp'].apply(lambda x: us_holidays.get(x, default=0))



holiday_idx = df_weather['IsHoliday'] != 0

df_weather.loc[holiday_idx, 'IsHoliday'] = 1

df_weather['IsHoliday'] = df_weather['IsHoliday'].astype(np.uint8)
BUILDING_METADATA_DTYPES = {'site_id': np.uint8, 'building_id': np.uint16, 'square_feet': np.float32, 'year_built': np.float32, 'floor_count': np.float32, 'building_eui': np.float32}

df_building_metadata = pd.read_csv('../input/ashrae-leaks/building_metadata_processed.csv', dtype=BUILDING_METADATA_DTYPES) # Using scraped building_metadata for site 0, 1, 2



print('Building Metadata Set Shape = {}'.format(df_building_metadata.shape))

print('Building Metadata Set Building Count = {}'.format(df_building_metadata['building_id'].nunique()))

print('Building Metadata Set Site Count = {}'.format(df_building_metadata['site_id'].nunique()))

print('Building Metadata Set Memory Usage = {:.2f} MB'.format(df_building_metadata.memory_usage().sum() / 1024**2))
df_building_count = df_building_metadata.groupby(['site_id'])['building_id'].count().sort_values(ascending=False)

df_building_count.index = [f'site_{site} ({count})' for site, count in df_building_count.to_dict().items()]



fig = plt.figure(figsize=(25, 8))

sns.barplot(x=df_building_count.index, y=df_building_count.values)



plt.xlabel('Sites', size=15, labelpad=20)

plt.ylabel('Building Counts', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=15)



plt.show()
df_building_counts = df_building_metadata.groupby(['site_id', 'primary_use'])['building_id'].count()

primary_use_idx = df_building_metadata['primary_use'].unique().tolist()

building_counts = {i: {primary_use: 0 for primary_use in primary_use_idx} for i in range(16)}

site_idx = [i for i in range(16)]  



for site in site_idx:

    for primary_use in primary_use_idx:

        try:

            count = df_building_counts[site][primary_use]

            building_counts[site][primary_use] = count 

        except KeyError:

            building_counts[site][primary_use] = 0



df_building_counts = pd.DataFrame(building_counts)



for col in df_building_counts.columns:

    df_building_counts[col] = df_building_counts[col].apply(lambda x: x / df_building_counts[col].sum() * 100)

    

for site in site_idx:

    df_building_counts[f'{site}_CumSum'] = df_building_counts[site].cumsum()

    

BAR_COUNT = np.arange(len(site_idx))  

BAR_WIDTH = 0.85

BAR_BOTTOMS = [f'{i}_CumSum' for i in range(16)]



plt.figure(figsize=(25, 12))



plt.bar(BAR_COUNT, df_building_counts.loc['Education', site_idx], color='tab:blue', edgecolor='white', width=BAR_WIDTH, label='Education')

plt.bar(BAR_COUNT, df_building_counts.loc['Lodging/residential', site_idx], bottom=df_building_counts.loc['Education', BAR_BOTTOMS], color='tab:orange', edgecolor='white', width=BAR_WIDTH, label='Lodging/residential')

plt.bar(BAR_COUNT, df_building_counts.loc['Office', site_idx], bottom=df_building_counts.loc['Lodging/residential', BAR_BOTTOMS], color='tab:green', edgecolor='white', width=BAR_WIDTH, label='Office')

plt.bar(BAR_COUNT, df_building_counts.loc['Entertainment/public assembly', site_idx], bottom=df_building_counts.loc['Office', BAR_BOTTOMS], color='tab:red', edgecolor='white', width=BAR_WIDTH, label='Entertainment/public assembly')

plt.bar(BAR_COUNT, df_building_counts.loc['Other', site_idx], color='tab:purple', bottom=df_building_counts.loc['Entertainment/public assembly', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Other')

plt.bar(BAR_COUNT, df_building_counts.loc['Retail', site_idx], color='tab:brown', bottom=df_building_counts.loc['Other', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Retail')

plt.bar(BAR_COUNT, df_building_counts.loc['Parking', site_idx], color='tab:pink', bottom=df_building_counts.loc['Retail', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Parking')

plt.bar(BAR_COUNT, df_building_counts.loc['Public services', site_idx], color='tab:gray', bottom=df_building_counts.loc['Parking', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Public services')

plt.bar(BAR_COUNT, df_building_counts.loc['Warehouse/storage', site_idx], color='tab:olive', bottom=df_building_counts.loc['Public services', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Warehouse/storage')

plt.bar(BAR_COUNT, df_building_counts.loc['Food sales and service', site_idx], color='tab:cyan', bottom=df_building_counts.loc['Warehouse/storage', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Food sales and service')

plt.bar(BAR_COUNT, df_building_counts.loc['Religious worship', site_idx], color='black', bottom=df_building_counts.loc['Food sales and service', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Religious worship')

plt.bar(BAR_COUNT, df_building_counts.loc['Healthcare', site_idx], color='yellow', bottom=df_building_counts.loc['Religious worship', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Healthcare')

plt.bar(BAR_COUNT, df_building_counts.loc['Utility', site_idx], color='aqua', bottom=df_building_counts.loc['Healthcare', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Utility')

plt.bar(BAR_COUNT, df_building_counts.loc['Technology/science', site_idx], color='deeppink', bottom=df_building_counts.loc['Utility', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Technology/science')

plt.bar(BAR_COUNT, df_building_counts.loc['Manufacturing/industrial', site_idx], color='blue', bottom=df_building_counts.loc['Technology/science', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Manufacturing/industrial')

plt.bar(BAR_COUNT, df_building_counts.loc['Services', site_idx], color='lime', bottom=df_building_counts.loc['Manufacturing/industrial', BAR_BOTTOMS], edgecolor='white', width=BAR_WIDTH, label='Services')



plt.xlabel('Sites', size=15, labelpad=20)

plt.ylabel('Building Type Percentage', size=15, labelpad=20)

plt.xticks(BAR_COUNT, [f'site_{i}' for i in site_idx])    

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

plt.title('Building Type Distribution in Sites', size=18, y=1.05)   



plt.show() 
primary_use_map = {'Education': 1, 'Office': 2, 'Entertainment/public assembly': 3, 'Lodging/residential': 4,

                   'Public services': 5, 'Healthcare': 6, 'Other': 7, 'Parking': 8, 'Manufacturing/industrial': 9,

                   'Food sales and service': 10, 'Retail': 11, 'Warehouse/storage': 12, 'Services': 13, 

                   'Technology/science': 14, 'Utility': 15, 'Religious worship': 16}



df_building_metadata['primary_use'] = df_building_metadata['primary_use'].map(primary_use_map).astype(np.uint8)

df_building_metadata.drop(columns=['building_eui'], inplace=True)
%%time



TRAIN_DTYPES = {'building_id': np.uint16, 'meter': np.uint8, 'meter_reading': np.float32}

df_train = pd.read_csv('../input/ashrae-energy-prediction/train.csv', dtype=TRAIN_DTYPES)

df_train['timestamp'] = pd.to_datetime(df_train['timestamp'], infer_datetime_format=True)



print('Training Set Shape after merge = {}'.format(df_train.shape))

print('Training Set Memory Usage after merge = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
def filter2(building_id, meter, min_length, plot=False, verbose=False):

    if verbose:

        print("building_id: {}, meter: {}".format(building_id, meter))

    temp_df = df_train[(df_train['building_id'] == building_id) & (df_train['meter'] == meter)]        

    target = temp_df['meter_reading'].values

    

    splitted_target = np.split(target, np.where(target[1:] != target[:-1])[0] + 1)

    splitted_date = np.split(temp_df['timestamp'].values, np.where(target[1:] != target[:-1])[0] + 1)



    building_idx = []

    for i, x in enumerate(splitted_date):

        if len(x) > min_length:

            start = x[0]

            end = x[-1]

            value = splitted_target[i][0]

            idx = df_train.query('(@start <= timestamp <= @end) and meter_reading == @value and meter == @meter and building_id == @building_id', engine='python').index.tolist()

            building_idx.extend(idx)

            if verbose:

                print('Length: {},\t{}  -  {},\tvalue: {}'.format(len(x), start, end, value))

                

    building_idx = pd.Int64Index(building_idx)

    if plot:

        fig, axes = plt.subplots(nrows=2, figsize=(16, 18), dpi=100)

        

        temp_df.set_index('timestamp')['meter_reading'].plot(ax=axes[0])     

        temp_df.drop(building_idx, axis=0).set_index('timestamp')['meter_reading'].plot(ax=axes[1])

        

        axes[0].set_title(f'Building {building_id} raw meter readings')

        axes[1].set_title(f'Building {building_id} filtered meter readings')

        

        plt.show()

        

    return building_idx

        
df_train['IsFiltered'] = 0



#################### SITE 0 ####################



##### METER 0 #####

print('[Site 0 - Electricity] Filtering leading constant values')



leading_zeros = defaultdict(list)

for building_id in range(105):

    leading_zeros_last_date = df_train.query('building_id == @building_id and meter_reading == 0 and timestamp < "2016-09-01 01:00:00"', engine='python')['timestamp'].max()

    leading_zeros[leading_zeros_last_date].append(building_id)



for timestamp in leading_zeros.keys():

    building_ids = leading_zeros[pd.Timestamp(timestamp)]

    filtered_idx = df_train[df_train['building_id'].isin(building_ids)].query('meter == 0 and timestamp <= @timestamp').index

    df_train.loc[filtered_idx, 'IsFiltered'] = 1



print('[Site 0 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 0 and meter == 0 and (meter_reading > 400 or meter_reading < -400)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 18 and meter == 0 and meter_reading < 1300').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 22 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 25 and meter == 0 and meter_reading <= 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 38 and meter == 0 and (meter_reading > 2000 or meter_reading < 0)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 41 and meter == 0 and (meter_reading > 2000 or meter_reading < 0)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 53 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 77 and meter == 0 and (meter_reading > 1000 or meter_reading < 0)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 78 and meter == 0 and (meter_reading > 20000 or meter_reading < 0)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 86 and meter == 0 and (meter_reading > 1000 or meter_reading < 0)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 101 and meter == 0 and meter_reading > 400').index, 'IsFiltered'] = 1



##### METER 1 #####

print('[Site 0 - Chilled Water] Filtering leading constant values')

site0_meter1_thresholds = {

    50: [7, 9, 43, 60, 75, 95, 97, 98]

}



for threshold in site0_meter1_thresholds:

    for building_id in site0_meter1_thresholds[threshold]:

        filtered_idx = filter2(building_id, 1, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



print('[Site 0 - Chilled Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 60 and meter == 1 and meter_reading > 25000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 103 and meter == 1 and meter_reading > 5000').index, 'IsFiltered'] = 1



#################### SITE 1 #####################



##### METER 0 #####

print('[Site 1 - Electricity] Filtering leading constant values')

site1_meter0_thresholds = {

    20: [106],

    50: [105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 119, 120, 127, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 149, 152, 155]

}



for threshold in site1_meter0_thresholds:

    for building_id in site1_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



##### METER 3 #####

print('[Site 1 - Hot Water] Filtering leading constant values')

site1_meter3_thresholds = {

    40: [106, 109, 112, 113, 114, 117, 119, 121, 138, 139, 144, 145]    

}



for threshold in site1_meter3_thresholds:

    for building_id in site1_meter3_thresholds[threshold]:

        filtered_idx = filter2(building_id, 3, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



print('[Site 1 - Hot Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 119 and meter == 3 and meter_reading > 4000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 121 and meter == 3 and meter_reading > 20000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 144 and meter == 3 and meter_reading > 100').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 145 and meter == 3 and meter_reading > 2000').index, 'IsFiltered'] = 1



#################### SITE 2 #####################



##### METER 0 #####

print('[Site 2 - Electricity] Filtering leading constant values')

site2_meter0_thresholds = {

    40: [278],

    100: [177, 258, 269],

    1000: [180]

}



for threshold in site2_meter0_thresholds:

    for building_id in site2_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

##### METER 1 #####

print('[Site 2 - Chilled Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 187 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 254 and meter == 1 and meter_reading > 1600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 257 and meter == 1 and meter_reading > 800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 273 and meter == 1 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 281 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1



print('[Site 2 - Chilled Water] Filtering leading constant values')

site2_meter1_thresholds = {

    100: [207],

    1000: [260]

}



for threshold in site2_meter1_thresholds:

    for building_id in site2_meter1_thresholds[threshold]:

        filtered_idx = filter2(building_id, 1, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



#################### SITE 3 #####################



##### METER 0 #####

print('[Site 3 - Electricity] Filtering leading constant values')

site3_meter0_thresholds = {

    100: [545]

}



for threshold in site3_meter0_thresholds:

    for building_id in site3_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



#################### SITE 4 #####################



##### METER 0 #####

print('[Site 4 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 592 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 609 and meter == 0 and meter_reading < 300').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 620 and meter == 0 and meter_reading < 750').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 626 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 627 and meter == 0 and meter_reading < 85').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 632 and meter == 0 and meter_reading < 30').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 645 and meter == 0 and meter_reading < 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 654 and meter == 0 and meter_reading < 40').index, 'IsFiltered'] = 1



print('[Site 4 - Electricity] Filtering leading constant values')

site4_meter0_thresholds = {

    100: [577, 604]

}



for threshold in site4_meter0_thresholds:

    for building_id in site4_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



#################### SITE 5 #####################



##### METER 0 #####

print('[Site 5 - Electricity] Filtering leading constant values')

site5_meter0_thresholds = {

    100: [681, 723, 733, 739],

    1000: [693]

}



for threshold in site5_meter0_thresholds:

    for building_id in site5_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



#################### SITE 6 #####################



##### METER 0 #####

print('[Site 6 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 749 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 758 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 769 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 770 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 773 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 778 and meter == 0 and meter_reading < 50').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 781 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 785 and meter == 0 and meter_reading < 200').index, 'IsFiltered'] = 1



##### METER 1 #####

print('[Site 6 - Chilled Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 745 and meter == 1 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 750 and meter == 1 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 753 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 755 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 765 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 769 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 770 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 771 and meter == 1 and meter_reading > 20').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 776 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 777 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 780 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 786 and meter == 1 and meter_reading > 7000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 787 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1



print('[Site 6 - Chilled Water] Filtering leading constant values')

site6_meter1_thresholds = {

    100: [748, 750, 752, 763, 767, 776, 786]

}



for threshold in site6_meter1_thresholds:

    for building_id in site6_meter1_thresholds[threshold]:

        filtered_idx = filter2(building_id, 1, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

##### METER 2 #####

print('[Site 6 - Steam] Filtering outliers')

df_train.loc[df_train.query('building_id == 762 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 764 and meter == 2 and meter_reading < 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 769 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 776 and meter == 2 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 784 and meter == 2 and meter_reading < 2000').index, 'IsFiltered'] = 1



print('[Site 6 - Steam] Filtering leading constant values')

site6_meter2_thresholds = {

    150: [750, 751, 753, 770],

    500: [774]

}



for threshold in site6_meter2_thresholds:

    for building_id in site6_meter2_thresholds[threshold]:

        filtered_idx = filter2(building_id, 2, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



#################### SITE 7 #####################



##### METER 0 #####

print('[Site 7 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 800 and meter == 0 and meter_reading < 75').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 801 and meter == 0 and meter_reading < 3000').index, 'IsFiltered'] = 1



print('[Site 7 - Electricity] Filtering leading constant values')

site7_meter0_thresholds = {

    100: [799, 800, 802]

}



for threshold in site7_meter0_thresholds:

    for building_id in site7_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

##### METER 1 #####

print('[Site 7 - Chilled Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 799 and meter == 1 and meter_reading > 4000 and timestamp > "2016-11-01"').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 800 and meter == 1 and meter_reading > 400 and timestamp > "2016-11-01"').index, 'IsFiltered'] = 1



print('[Site 7 - Chilled Water] Filtering leading constant values')

site7_meter1_thresholds = {

    50: [789, 790, 792]

}



for threshold in site7_meter1_thresholds:

    for building_id in site7_meter1_thresholds[threshold]:

        filtered_idx = filter2(building_id, 1, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

##### METER 2 #####

print('[Site 7 - Steam] Filtering outliers')

df_train.loc[df_train.query('building_id == 797 and meter == 2 and meter_reading > 8000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 800 and meter == 2 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 803 and meter == 2 and meter_reading == 0').index, 'IsFiltered'] = 1



print('[Site 7 - Steam] Filtering leading constant values')

site7_meter2_thresholds = {

    100: [800]

}



for threshold in site7_meter2_thresholds:

    for building_id in site7_meter2_thresholds[threshold]:

        filtered_idx = filter2(building_id, 2, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



#################### SITE 8 #####################



##### METER 0 #####

print('[Site 8 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 857 and meter == 0 and meter_reading > 10').index, 'IsFiltered'] = 1



print('[Site 8 - Electricity] Filtering leading constant values')

site8_meter0_thresholds = {

    1000: [815, 848]

}



for threshold in site8_meter0_thresholds:

    for building_id in site8_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



#################### SITE 9 #####################



##### METER 0 #####

print('[Site 9 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 886 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 904 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 921 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 927 and meter == 0 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 927 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 949 and meter == 0 and meter_reading > 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 954 and meter == 0 and meter_reading > 10000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 955 and meter == 0 and meter_reading > 4000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 956 and meter == 0 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 961 and meter == 0 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 962 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 966 and meter == 0 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 977 and meter == 0 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 983 and meter == 0 and meter_reading > 3500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 986 and meter == 0 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 990 and meter == 0 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 993 and meter == 0 and meter_reading > 6000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 997 and meter == 0 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)



print('[Site 9 - Electricity] Filtering leading constant values')

site9_meter0_thresholds = {

    40: [897],

    50: [874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 898, 899,

         900, 901, 902, 903, 904, 905, 906, 907, 908, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925,

         926, 927, 928, 929, 930, 931, 932, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952,

         953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977,

         978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997],

    100: [909],

}



for threshold in site9_meter0_thresholds:

    for building_id in site9_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



##### METER 1 #####

print('[Site 9 - Chilled Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 879 and meter == 1 and meter_reading > 8000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 880 and meter == 1 and meter_reading > 8000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 885 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 891 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 900 and meter == 1 and meter_reading > 5000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 903 and meter == 1 and meter_reading > 20000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 906 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 907 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 908 and meter == 1 and meter_reading > 2500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 910 and meter == 1 and meter_reading > 4000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 920 and meter == 1 and meter_reading > 4000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 923 and meter == 1 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 925 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 926 and meter == 1 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 927 and meter == 1 and meter_reading > 20000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 929 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 931 and meter == 1 and meter_reading > 6000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 934 and meter == 1 and meter_reading > 2500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 946 and meter == 1 and meter_reading > 10000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 948 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 949 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 954 and meter == 1 and meter_reading > 50000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 955 and meter == 1 and meter_reading > 15000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 956 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 957 and meter == 1 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 961 and meter == 1 and meter_reading > 8000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 963 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 964 and meter == 1 and meter_reading > 5000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 965 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 967 and meter == 1 and meter_reading > 1750').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 969 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 973 and meter == 1 and meter_reading > 2500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 976 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 978 and meter == 1 and meter_reading > 4000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 983 and meter == 1 and meter_reading > 3800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 989 and meter == 1 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 990 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 993 and meter == 1 and meter_reading > 10000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 994 and meter == 1 and meter_reading > 2500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 996 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)



print('[Site 9 - Chilled Water] Filtering leading constant values')

site9_meter1_thresholds = {

    50: [874, 875, 879, 880, 883, 885, 886, 887, 888, 889, 890, 891, 893, 894, 895, 896, 898, 899, 900, 901, 903, 905, 906, 907, 908,

         910, 911, 912, 913, 914, 915, 916, 917, 918, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 931, 932, 933, 934, 935, 942,

         945, 946, 948, 949, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 971, 972,

         973, 974, 975, 976, 978, 979, 980, 981, 983, 987, 989, 990, 991, 992, 993, 994, 995, 996, 997]

}



for threshold in site9_meter1_thresholds:

    for building_id in site9_meter1_thresholds[threshold]:

        filtered_idx = filter2(building_id, 1, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



##### METER 2 #####

print('[Site 9 - Steam] Filtering outliers')

df_train.loc[df_train.query('building_id == 875 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 876 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 878 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 879 and meter == 2 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 880 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 880 and meter == 2 and meter_reading > 600 and ("2016-06-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 885 and meter == 2 and meter_reading > 250').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 886 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 886 and meter == 2 and meter_reading > 300 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 887 and meter == 2 and meter_reading > 325').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 888 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 889 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 890 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 894 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 895 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 896 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 896 and meter == 2 and meter_reading > 400 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 898 and meter == 2 and meter_reading > 150').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 898 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 899 and meter == 2 and meter_reading > 800 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 900 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 901 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 903 and meter == 2 and meter_reading > 2400').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 905 and meter == 2 and meter_reading > 140').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 906 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 907 and meter == 2 and meter_reading > 300').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 907 and meter == 2 and meter_reading > 200 and ("2016-07-01" < timestamp < "2016-08-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 908 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 908 and meter == 2 and meter_reading > 300 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 910 and meter == 2 and meter_reading > 300 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 911 and meter == 2 and meter_reading > 1250').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 912 and meter == 2 and meter_reading > 120').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 912 and meter == 2 and meter_reading > 90 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 913 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 914 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 915 and meter == 2 and meter_reading > 750').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 915 and meter == 2 and meter_reading > 185 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 916 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 917 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 918 and meter == 2 and meter_reading > 300').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 920 and meter == 2 and meter_reading > 490').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 921 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 922 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 922 and meter == 2 and meter_reading > 200 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 924 and meter == 2 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 926 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 926 and meter == 2 and meter_reading > 70 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 927 and meter == 2 and meter_reading > 4000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 928 and meter == 2 and meter_reading > 1250').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 929 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 931 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 931 and meter == 2 and meter_reading > 400 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 932 and meter == 2 and meter_reading > 1750').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 933 and meter == 2 and meter_reading > 75').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 934 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 942 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 945 and meter == 2 and meter_reading > 1200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 946 and meter == 2 and meter_reading > 1200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 948 and meter == 2 and meter_reading > 120').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 949 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 951 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 952 and meter == 2 and meter_reading > 1600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 953 and meter == 2 and meter_reading > 1250').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 954 and meter == 2 and meter_reading > 10000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 955 and meter == 2 and meter_reading > 1750').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 956 and meter == 2 and meter_reading > 350').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 957 and meter == 2 and meter_reading > 1200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 957 and meter == 2 and meter_reading > 350 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 958 and meter == 2 and meter_reading > 1200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 959 and meter == 2 and meter_reading > 2500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 960 and meter == 2 and meter_reading > 350').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 960 and meter == 2 and meter_reading > 100 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 961 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 963 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 964 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 965 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 966 and meter == 2 and meter_reading > 1750').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 966 and meter == 2 and meter_reading > 500 and ("2016-07-01" < timestamp < "2016-08-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 967 and meter == 2 and meter_reading > 575').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 968 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 969 and meter == 2 and meter_reading > 700').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 969 and meter == 2 and meter_reading > 400 and ("2016-07-01" < timestamp < "2016-08-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 971 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 972 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 973 and meter == 2 and meter_reading > 450').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 974 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 976 and meter == 2 and meter_reading > 60').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 978 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 978 and meter == 2 and meter_reading > 1250 and ("2016-07-01" < timestamp < "2016-08-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 980 and meter == 2 and meter_reading > 150').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 981 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 983 and meter == 2 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 983 and meter == 2 and meter_reading > 1750 and ("2016-05-01" < timestamp < "2016-08-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 987 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 989 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 991 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 991 and meter == 2 and meter_reading > 200 and ("2016-06-01" < timestamp < "2016-08-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 992 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 993 and meter == 2 and meter_reading > 6000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 994 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 995 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 996 and meter == 2 and meter_reading > 260').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 997 and meter == 2 and meter_reading > 300').index, 'IsFiltered'] = 1

df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)



print('[Site 9 - Steam] Filtering leading constant values')

site9_meter2_thresholds = {

    40: [889, 910, 934, 955, 962, 974, 976],

    50: [874, 875, 876, 878, 879, 880, 885, 886, 887, 888, 890, 894, 895, 896, 898, 901, 903, 905, 906, 907, 908, 911, 912, 913, 914, 915, 916, 917, 918, 920, 921,

         922, 924, 925, 926, 927, 928, 929, 931, 932, 933, 942, 945, 946, 948, 949, 951, 952, 953, 954, 956, 957, 958, 959, 960, 961, 963, 964, 965, 966, 967, 968, 

         969, 971, 972, 973, 978, 979, 980, 981, 983, 987, 989, 991, 992, 993, 994, 995, 996, 997]

}



for threshold in site9_meter2_thresholds:

    for building_id in site9_meter2_thresholds[threshold]:

        filtered_idx = filter2(building_id, 2, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

#################### SITE 10 #####################



##### METER 0 #####           

print('[Site 10 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 998 and meter == 0 and meter_reading > 300').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1000 and meter == 0 and meter_reading > 300').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1006 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1008 and meter == 0 and (meter_reading > 250 or meter_reading < 5)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1019 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1025 and meter == 0 and meter_reading < 20').index, 'IsFiltered'] = 1



##### METER 1 #####

print('[Site 10 - Chilled Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 1003 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1017 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1



##### METER 3 #####

print('[Site 10 - Hot Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 1000 and meter == 3 and meter_reading > 5000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1003 and meter == 3 and meter_reading > 40 and ("2016-06-01" < timestamp < "2016-08-01")').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1012 and meter == 3 and meter_reading > 500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1017 and meter == 3 and meter_reading > 5000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1018 and meter == 3 and meter_reading > 10000').index, 'IsFiltered'] = 1

        

#################### SITE 12 #####################



##### METER 0 #####        

print('[Site 12 - Electricity] Filtering leading constant values')

site12_meter0_thresholds = {

    50: [1066]

}



for threshold in site12_meter0_thresholds:

    for building_id in site12_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

#################### SITE 13 #####################



##### METER 0 #####

print('[Site 13 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 1070 and meter == 0 and (meter_reading < 0 or meter_reading > 200)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1073 and meter == 0 and (meter_reading < 60 or meter_reading > 800)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1073 and meter == 0 and timestamp < "2016-08-01" and meter_reading > 200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1082 and meter == 0 and (meter_reading < 5 or meter_reading > 200)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1088 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1098 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1100 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1119 and meter == 0 and meter_reading < 30').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1128 and meter == 0 and (meter_reading < 25 or meter_reading > 175)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1153 and meter == 0 and (meter_reading < 90 or meter_reading > 250)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1157 and meter == 0 and (meter_reading < 110 or meter_reading > 200)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1162 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1163 and meter == 0 and meter_reading < 30').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1165 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1167 and meter == 0 and (meter_reading == 0 or meter_reading > 250)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1168 and meter == 0 and meter_reading > 6000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1172 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1177 and meter == 0 and meter_reading < 15').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1185 and meter == 0 and (meter_reading > 300 or meter_reading < 10)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1202 and meter == 0 and (meter_reading < 50 or meter_reading > 300)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1203 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1206 and meter == 0 and meter_reading < 40').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1207 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1208 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1209 and meter == 0 and (meter_reading > 350 or meter_reading < 175)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1210 and meter == 0 and (meter_reading > 225 or meter_reading < 75)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1212 and meter == 0 and meter_reading < 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1219 and meter == 0 and (meter_reading < 35 or meter_reading > 300)').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1222 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1



print('[Site 13 - Electricity] Filtering leading constant values')

site13_meter0_thresholds = {

    40: [1079, 1096, 1113, 1154, 1160, 1169, 1170, 1189, 1221]

}



for threshold in site13_meter0_thresholds:

    for building_id in site13_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



##### METER 1 #####

print('[Site 13 - Chilled Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 1088 and meter == 1 and meter_reading > 10000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1220 and meter == 1 and meter_reading > 4000').index, 'IsFiltered'] = 1



print('[Site 13 - Chilled Water] Filtering leading constant values')

site13_meter1_thresholds = {

    40: [1130, 1160]

}



for threshold in site13_meter1_thresholds:

    for building_id in site13_meter1_thresholds[threshold]:

        filtered_idx = filter2(building_id, 1, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

##### METER 2 #####

print('[Site 13 - Steam] Filtering outliers')

df_train.loc[df_train.query('building_id == 1075 and meter == 2 and meter_reading > 3500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1099 and meter == 2 and meter_reading > 30000').index, 'IsFiltered'] = 1



print('[Site 13 - Steam] Filtering leading constant values')

site13_meter2_thresholds = {

    40: [1072, 1098, 1158],

    100: [1111],

    500: [1129, 1176, 1189]

}



for threshold in site13_meter2_thresholds:

    for building_id in site13_meter2_thresholds[threshold]:

        filtered_idx = filter2(building_id, 2, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

#################### SITE 14 #####################



##### METER 0 #####

print('[Site 14 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 1252 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1258 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1263 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1284 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1



print('[Site 14 - Electricity] Filtering leading constant values')

site14_meter0_thresholds = {

    100: [1223, 1225, 1226, 1240, 1241, 1250, 1255, 1264, 1265, 1272, 1275, 1276, 1277, 1278, 1279, 1280, 1283,

          1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1302, 1303, 1317, 1322],

    300: [1319],

    500: [1233, 1234]

}



for threshold in site14_meter0_thresholds:

    for building_id in site14_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



##### METER 1 #####

print('[Site 14 - Chilled Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 1236 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1242 and meter == 1 and meter_reading > 800').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1276 and meter == 1 and meter_reading > 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1280 and meter == 1 and meter_reading > 120').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1311 and meter == 1 and meter_reading > 1100').index, 'IsFiltered'] = 1



print('[Site 14 - Chilled Water] Filtering leading constant values')

site14_meter1_thresholds = {

    50: [1239, 1245, 1247, 1248, 1254, 1287, 1295, 1307, 1308],

    100: [1225, 1226, 1227, 1230, 1232, 1233, 1234, 1237, 1240, 1246, 1260, 1263, 1264, 1272, 1276, 

          1280, 1288, 1290, 1291, 1292, 1293, 1294, 1296, 1297, 1300, 1302, 1303, 1310, 1311, 1312, 

          1317],

    500: [1223]

}



for threshold in site14_meter1_thresholds:

    for building_id in site14_meter1_thresholds[threshold]:

        filtered_idx = filter2(building_id, 1, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

##### METER 2 #####

print('[Site 14 - Steam] Filtering outliers')

df_train.loc[df_train.query('building_id == 1249 and meter == 2 and meter_reading > 4000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1254 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1256 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1263 and meter == 2 and meter_reading > 9000').index, 'IsFiltered'] = 1



print('[Site 14 - Steam] Filtering leading constant values')

site14_meter2_thresholds = {

    50: [1225, 1226, 1239, 1254, 1284, 1285, 1286, 1287, 1289, 1290,

         1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1301,

         1303, 1305, 1308, 1309, 1310],

    100: [1238, 1243, 1245, 1247, 1248, 1249, 1250, 1263, 1307]

}



for threshold in site14_meter2_thresholds:

    for building_id in site14_meter2_thresholds[threshold]:

        filtered_idx = filter2(building_id, 2, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

##### METER 3 #####

print('[Site 14 - Hot Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 1231 and meter == 3 and meter_reading > 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1232 and meter == 3 and meter_reading > 4500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1236 and meter == 3 and meter_reading > 600').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1251 and meter == 3 and meter_reading > 3000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1268 and meter == 3 and meter_reading > 1500').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1274 and meter == 3 and meter_reading > 1000').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1298 and meter == 3 and meter_reading > 5000').index, 'IsFiltered'] = 1



print('[Site 14 - Hot Water] Filtering leading constant values')

site14_meter3_thresholds = {

    40: [1270, 1322, 1323],

    50: [1223, 1224, 1227, 1228, 1229, 1231, 1233, 1234, 1235, 1236, 1240, 1242, 1244, 1246, 1251, 1252, 1260, 1262, 1265,

         1266, 1267, 1269, 1271, 1272, 1273, 1274, 1275, 1276, 1312, 1317, 1318, 1319, 1321],

    100: [1231, 1232, 1237]

}



for threshold in site14_meter3_thresholds:

    for building_id in site14_meter3_thresholds[threshold]:

        filtered_idx = filter2(building_id, 3, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

#################### SITE 15 #####################



##### METER 0 #####        

print('[Site 15 - Electricity] Filtering outliers')

df_train.loc[df_train.query('building_id == 1383 and meter == 0 and meter_reading < 60').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1401 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1414 and meter == 0 and meter_reading < 30').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1423 and meter == 0 and meter_reading < 5').index, 'IsFiltered'] = 1



print('[Site 15 - Electricity] Filtering leading constant values')

site15_meter0_thresholds = {

    50: [1345, 1359, 1446]

}



for threshold in site15_meter0_thresholds:

    for building_id in site15_meter0_thresholds[threshold]:

        filtered_idx = filter2(building_id, 0, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



##### METER 1 #####

print('[Site 15 - Chilled Water] Filtering outliers')

df_train.loc[df_train.query('building_id == 1349 and meter == 1 and meter_reading > 1000 and timestamp < "2016-04-15"').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1382 and meter == 1 and meter_reading > 300 and timestamp < "2016-02-01"').index, 'IsFiltered'] = 1



print('[Site 15 - Chilled Water] Filtering leading constant values')

site15_meter1_thresholds = {

    50: [1363, 1410]

}



for threshold in site15_meter1_thresholds:

    for building_id in site15_meter1_thresholds[threshold]:

        filtered_idx = filter2(building_id, 1, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1

        

##### METER 2 #####        

print('[Site 15 - Steam] Filtering outliers')

df_train.loc[df_train.query('building_id == 1355 and meter == 2 and meter_reading < 200').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1361 and meter == 2 and meter_reading < 203').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1392 and meter == 2 and meter_reading < 150').index, 'IsFiltered'] = 1

df_train.loc[df_train.query('building_id == 1426 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1



print('[Site 15 - Steam] Filtering leading constant values')

site15_meter2_thresholds = {

    20: [1425, 1427],

    40: [1329, 1337, 1338, 1341, 1342, 1344, 1347, 1350, 1351, 1354, 1360, 1364,

         1367, 1377, 1379, 1381, 1382, 1383, 1391, 1396, 1402, 1405, 1406, 1409,

         1414, 1418, 1424, 1430, 1433, 1437]

}



for threshold in site15_meter2_thresholds:

    for building_id in site15_meter2_thresholds[threshold]:

        filtered_idx = filter2(building_id, 2, threshold)

        df_train.loc[filtered_idx, 'IsFiltered'] = 1



df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)

df_train.drop(columns=['IsFiltered'], inplace=True)
%%time



TEST_DTYPES = {'building_id': np.uint16, 'meter': np.uint8}

df_test = pd.read_csv('../input/ashrae-energy-prediction/test.csv', dtype=TEST_DTYPES)

df_test['timestamp'] = pd.to_datetime(df_test['timestamp'], infer_datetime_format=True)

df_test.drop(columns=['row_id'], inplace=True)



df_all = pd.concat([df_train, df_test], ignore_index=True, sort=False)

df_all = df_all.merge(df_building_metadata, on=['building_id'], how='left')

df_all = df_all.merge(df_weather, on=['site_id', 'timestamp'], how='left')



print('Training Set Shape after merge = {}'.format(df_train.shape))

print('Test Set Shape after merge = {}'.format(df_test.shape))

print('Training Set Memory Usage after merge = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))

print('Test Set Memory Usage after merge = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))



del df_train, df_test

gc.collect()
# df_all['air_temperature_int'] = df_all['air_temperature'].astype(np.int8)

# df_all['dew_temperature_int'] = df_all['dew_temperature'].astype(np.int8)



# df_all['LogTargetMeanOnAirTemperature'] = np.log1p(df_all.groupby(['building_id', 'meter', 'air_temperature_int'])['meter_reading'].transform('mean').values)

# df_all['LogTargetMeanOnDewTemperature'] = np.log1p(df_all.groupby(['building_id', 'meter', 'dew_temperature_int'])['meter_reading'].transform('mean').values)



# df_all.drop(columns=['air_temperature_int', 'dew_temperature_int'], inplace=True)
df_all['Building_Reading_Count'] = df_all['building_id'].map(df_all['building_id'].value_counts(dropna=False))

df_all['Building_Reading_Count'] = df_all['Building_Reading_Count'].astype(np.uint32)



df_all['Site_Reading_Count'] = df_all['site_id'].map(df_all['site_id'].value_counts(dropna=False))

df_all['Site_Reading_Count'] = df_all['Site_Reading_Count'].astype(np.uint32)



df_all['Building_MeterType_Reading_Count'] = df_all.groupby(['building_id', 'meter'])['building_id'].transform('count')

df_all['Building_MeterType_Reading_Count'] = df_all['Building_MeterType_Reading_Count'].astype(np.uint16)



df_all['Site_BuildingType_Reading_Count'] = df_all.groupby(['site_id', 'primary_use'])['site_id'].transform('count')

df_all['Site_BuildingType_Reading_Count'] = df_all['Site_BuildingType_Reading_Count'].astype(np.uint16)



df_all['Building_MeterType_Reading_Percentage'] = df_all['Building_MeterType_Reading_Count'] / df_all['Building_Reading_Count']

df_all['Building_MeterType_Reading_Percentage'] = df_all['Building_MeterType_Reading_Percentage'].astype(np.float32)



df_all['Site_BuildingType_Reading_Percentage'] = df_all['Site_BuildingType_Reading_Count'] / df_all['Site_Reading_Count']

df_all['Site_BuildingType_Reading_Percentage'] = df_all['Site_BuildingType_Reading_Percentage'].astype(np.float32)
%%time



SPLIT_DATE = '2017-01-01 00:00:00'



# Dropping merged columns for size optimization

WEATHER_COLS = ['site_id', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed', 

                'HourOfDay', 'DayOfWeek', 'humidity', 'IsHoliday']

BUILDING_COLS = ['square_feet', 'year_built', 'floor_count', 'primary_use']

ALL_DROP_COLS = WEATHER_COLS + BUILDING_COLS

df_all.drop(columns=ALL_DROP_COLS, inplace=True)



df_all[df_all['timestamp'] < SPLIT_DATE].to_pickle('train.pkl')

df_all[df_all['timestamp'] >= SPLIT_DATE].to_pickle('test.pkl')

df_weather.to_pickle('weather.pkl')

df_building_metadata.to_pickle('building_metadata.pkl')



print('Processed Training Set Shape = {}'.format(df_all[df_all['timestamp'] < SPLIT_DATE].shape))

print('Processed Test Set Shape = {}'.format(df_all[df_all['timestamp'] >= SPLIT_DATE].shape))

print('Processed Training Set Memory Usage = {:.2f} MB'.format(df_all[df_all['timestamp'] < SPLIT_DATE].memory_usage().sum() / 1024**2))

print('Processed Test Set Memory Usage = {:.2f} MB'.format(df_all[df_all['timestamp'] >= SPLIT_DATE].memory_usage().sum() / 1024**2))

print('Processed Weather Set Shape = {}'.format(df_weather.shape))

print('Processed Weather Set Memory Usage = {:.2f} MB'.format(df_weather.memory_usage().sum() / 1024**2))

print('Processed Building Metadata Set Shape = {}'.format(df_building_metadata.shape))

print('Processed Building Metadata Set Memory Usage = {:.2f} MB'.format(df_building_metadata.memory_usage().sum() / 1024**2))