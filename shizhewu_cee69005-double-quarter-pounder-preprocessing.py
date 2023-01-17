import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import warnings

warnings.filterwarnings('ignore')

import lightgbm as lgb

import gc

import holidays
building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
weather_train["datetime"] = pd.to_datetime(weather_train["timestamp"])

weather_train["month"] = weather_train["datetime"].dt.month

weather_train = weather_train.set_index(['site_id','month'])
air_temperature_filler = pd.DataFrame(weather_train.groupby(['site_id','month'])['air_temperature'].mean(),columns = ["air_temperature"])

dew_temperature_filler = pd.DataFrame(weather_train.groupby(['site_id','month'])['dew_temperature'].mean(),columns = ["dew_temperature"])

wind_direction_filler =  pd.DataFrame(weather_train.groupby(['site_id','month'])['wind_direction'].mean(),columns = ['wind_direction'])

wind_speed_filler =  pd.DataFrame(weather_train.groupby(['site_id','month'])['wind_speed'].mean(),columns = ['wind_speed'])
cloud_coverage_filler = weather_train.groupby(['site_id','month'])['cloud_coverage'].mean()

cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method = 'ffill'),columns = ["cloud_coverage"])

precip_depth_filler = weather_train.groupby(['site_id','month'])['precip_depth_1_hr'].mean()

precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method = 'ffill'),columns = ['precip_depth_1_hr'])

sea_level_filler = weather_train.groupby(['site_id','month'])['sea_level_pressure'].mean()

sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method = 'ffill'),columns = ['sea_level_pressure'])
order = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'timestamp','datetime']

weather_train = weather_train[order]

filler = pd.concat([air_temperature_filler,cloud_coverage_filler,dew_temperature_filler,precip_depth_filler,sea_level_filler,wind_direction_filler,wind_speed_filler],axis = 1)

weather_train.update(filler,overwrite = False)

weather_train = weather_train.reset_index()

weather_train = weather_train.drop(['datetime','month'],axis = 1)

weather_train.to_csv("weather_train_filled.csv")
merge1 = building.merge(train, left_on = 'building_id', right_on = 'building_id')

data_table = merge1.merge(weather_train, left_on = ['site_id','timestamp'], right_on = ['site_id','timestamp'])
data_table
# find holiday

en_holidays = holidays.England()

ir_holidays = holidays.Ireland()

ca_holidays = holidays.Canada()

us_holidays = holidays.UnitedStates()



en_idx = data_table.query('site_id == 1 or site_id == 5').index

ir_idx = data_table.query('site_id == 12').index

ca_idx = data_table.query('site_id == 7 or site_id == 11').index

us_idx = data_table.query('site_id == 0 or site_id == 2 or site_id == 3 or site_id == 4 or site_id == 6 or site_id == 8 or site_id == 9 or site_id == 10 or site_id == 13 or site_id == 14 or site_id == 15').index



data_table['IsHoliday'] = 0

data_table.loc[en_idx, 'IsHoliday'] = data_table.loc[en_idx, 'timestamp'].apply(lambda x: en_holidays.get(x, default = 0))

data_table.loc[ir_idx, 'IsHoliday'] = data_table.loc[ir_idx, 'timestamp'].apply(lambda x: ir_holidays.get(x, default = 0))

data_table.loc[ca_idx, 'IsHoliday'] = data_table.loc[ca_idx, 'timestamp'].apply(lambda x: ca_holidays.get(x, default = 0))

data_table.loc[us_idx, 'IsHoliday'] = data_table.loc[us_idx, 'timestamp'].apply(lambda x: us_holidays.get(x, default = 0))



holiday_idx = data_table['IsHoliday'] != 0

data_table.loc[holiday_idx, 'IsHoliday'] = 1

data_table['IsHoliday'] = data_table['IsHoliday'].astype(np.uint8)



# drop holiday row

data_table.drop(data_table.query('IsHoliday == 1').index, inplace = True)

data_table.drop(columns = ['IsHoliday'], inplace = True)
data_table
drop_features = ["primary_use", "cloud_coverage", "floor_count", "precip_depth_1_hr", "wind_speed","year_built"]

data_table.drop(drop_features, axis = 1, inplace = True) 
data_table.to_csv("data_table_1.csv")
df = pd.read_csv("../input/cee69005-double-quarter-pounder-preprocessing/data_table_1.csv", parse_dates = ["timestamp"], index_col = [0])

plt.figure(figsize = [12,6])

df[df['site_id'] == 0][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(alpha=0.8, color = 'tab:blue').set_ylabel('Mean meter reading', fontsize = 12)

plt.savefig('site0.png')
plt.figure(figsize = [12,6])

df[(df['meter'] == 2) & (df['building_id'] == 1099)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(alpha = 0.8, label = 'By hour', color = 'tab:blue').set_ylabel('Mean meter reading', fontsize = 12)

plt.savefig('building1099.png')
df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
def find_bad_sitezero(X):

    """Returns indices of bad rows from the early days of Site 0 (UCF)."""

    return X[(X.timestamp < 3378) & (X.site_id == 0) & (X.meter == 0)].index

def find_bad_building1099(X, y):

    """Returns indices of bad rows (with absurdly high readings) from building 1099."""

    return X[(X.building_id == 1099) & (X.meter == 2) & (y > 3e4)].index

def find_bad_rows(X, y):

    return find_bad_sitezero(X).union(find_bad_building1099(X, y))

bad_rows = find_bad_rows(df, df.meter_reading)

df = df.drop(index = bad_rows)

df
df.to_csv("final_data_table.csv")