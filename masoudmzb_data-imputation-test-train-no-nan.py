import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Import datasets For train

train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
weather_train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

# Import test
test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')
weather_test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

# Same for Both
building_meta_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage_2(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if col == 'timestamp': continue
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df
train_df = reduce_mem_usage_2(train_df ,use_float16=True)
weather_train_df = reduce_mem_usage_2(weather_train_df ,use_float16=True)
building_meta_df = reduce_mem_usage_2(building_meta_df ,use_float16=True)
test_df = reduce_mem_usage_2(test_df ,use_float16=True)
weather_test_df = reduce_mem_usage_2(weather_test_df ,use_float16=True)
train_df = train_df.merge(building_meta_df, on='building_id', how='left')
train_df = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

test_df = test_df.merge(building_meta_df, on='building_id', how='left')
test_df = test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')

del weather_train_df, building_meta_df, weather_test_df
gc.collect();
train_df.timestamp = pd.to_datetime(train_df.timestamp)
test_df.timestamp = pd.to_datetime(test_df.timestamp)
# in Service type of primary use we don't have any value for floor count and year_built
# we will fill it by site_id ===> Location_base
train_df[train_df.primary_use == 'Services'].isnull().sum() * 100 / train_df[train_df.primary_use == 'Services'].shape[0]

# we have 3 primary_use we don't have any value for floor count 
# Food sales and service | Religious worship | Services
mean_df = train_df.groupby('primary_use').year_built.agg(['mean']).to_dict()
for this_primary_use in train_df.primary_use.unique():
    if this_primary_use == 'Services':
        continue
    train_df.loc[train_df.primary_use == this_primary_use, ['year_built']] = train_df.loc[
        train_df.primary_use == this_primary_use, ['year_built']].fillna(mean_df['mean'][this_primary_use])

    
mean_df = test_df.groupby('primary_use').year_built.agg(['mean']).to_dict()
for this_primary_use in test_df.primary_use.unique():
    if this_primary_use == 'Services':
        continue
    test_df.loc[test_df.primary_use == this_primary_use, ['year_built']] = test_df.loc[
        test_df.primary_use == this_primary_use, ['year_built']].fillna(mean_df['mean'][this_primary_use])
# for those type of primary use which we don't have any year_built data. we can use mean of site id
mean_df_dict = train_df.groupby('site_id').year_built.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    train_df.loc[train_df.site_id == sid, ['year_built']] = train_df.loc[
        train_df.site_id == sid, ['year_built']].fillna(mean_df_dict['mean'][sid])
    
mean_df_dict = test_df.groupby('site_id').year_built.agg(['mean']).to_dict()
for sid in test_df.site_id.unique():
    test_df.loc[test_df.site_id == sid, ['year_built']] = test_df.loc[
        test_df.site_id == sid, ['year_built']].fillna(mean_df_dict['mean'][sid])

train_df.isnull().sum() * 100 / train_df.shape[0]
test_df.isnull().sum() * 100 / test_df.shape[0]
# Floor count
mean_of_floor_count_df = train_df.groupby('primary_use').floor_count.agg(['mean'])
mean_of_floor_count_df_dict = mean_of_floor_count_df.to_dict()


for this_primary_use in train_df.primary_use.unique():
    if this_primary_use == 'Services' or this_primary_use == 'Food sales and service' or this_primary_use == 'Religious worship':
        continue
    train_df.loc[train_df.primary_use == this_primary_use, ['floor_count']] = train_df.loc[
        train_df.primary_use == this_primary_use, ['floor_count']].fillna(mean_of_floor_count_df_dict['mean'][this_primary_use])
    
    
mean_of_floor_count_df = test_df.groupby('primary_use').floor_count.agg(['mean'])
mean_of_floor_count_df_dict = mean_of_floor_count_df.to_dict()


for this_primary_use in test_df.primary_use.unique():
    if this_primary_use == 'Services' or this_primary_use == 'Food sales and service' or this_primary_use == 'Religious worship':
        continue
    test_df.loc[test_df.primary_use == this_primary_use, ['floor_count']] = test_df.loc[
        test_df.primary_use == this_primary_use, ['floor_count']].fillna(mean_of_floor_count_df_dict['mean'][this_primary_use])
# for those type of primary use which we don't have any floor_count data. we can use mean of site id
mean_df_dict = train_df.groupby('site_id').floor_count.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    train_df.loc[train_df.site_id == sid, ['floor_count']] = train_df.loc[
        train_df.site_id == sid, ['floor_count']].fillna(mean_df_dict['mean'][sid])
    
mean_df_dict = test_df.groupby('site_id').floor_count.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    test_df.loc[test_df.site_id == sid, ['floor_count']] = test_df.loc[
        test_df.site_id == sid, ['floor_count']].fillna(mean_df_dict['mean'][sid])
test_df.isnull().sum() * 100 / test_df.shape[0]
# for i in train_df.site_id.unique():
#     print(i)

# train_df.cloud_coverage.mean()
mean_of_cloud_coverage_df_dict = train_df.groupby('site_id').cloud_coverage.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    if sid == 7 or sid == 11 :
        continue
    train_df.loc[train_df.site_id == sid, ['cloud_coverage']] = train_df.loc[
        train_df.site_id == sid, ['cloud_coverage']].fillna(mean_of_cloud_coverage_df_dict['mean'][sid])

mean_dict = test_df.groupby('site_id').cloud_coverage.agg(['mean']).to_dict()
for sid in test_df.site_id.unique():
    if sid == 7 or sid == 11 :
        continue
    test_df.loc[test_df.site_id == sid, ['cloud_coverage']] = test_df.loc[
        test_df.site_id == sid, ['cloud_coverage']].fillna(mean_dict['mean'][sid])
test_df.isnull().sum() * 100 / test_df.shape[0]
mean_df_dict = train_df.groupby('site_id').wind_speed.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    train_df.loc[train_df.site_id == sid, ['wind_speed']] = train_df.loc[
        train_df.site_id == sid, ['wind_speed']].fillna(mean_df_dict['mean'][sid])
    
    
mean_df_dict = train_df.groupby('site_id').wind_direction.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    train_df.loc[train_df.site_id == sid, ['wind_direction']] = train_df.loc[
        train_df.site_id == sid, ['wind_direction']].fillna(mean_df_dict['mean'][sid])
    
mean_df_dict = train_df.groupby('site_id').dew_temperature.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    train_df.loc[train_df.site_id == sid, ['dew_temperature']] = train_df.loc[
        train_df.site_id == sid, ['dew_temperature']].fillna(mean_df_dict['mean'][sid])
    
mean_df_dict = train_df.groupby('site_id').air_temperature.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    train_df.loc[train_df.site_id == sid, ['air_temperature']] = train_df.loc[
        train_df.site_id == sid, ['air_temperature']].fillna(mean_df_dict['mean'][sid])
mean_df_dict = test_df.groupby('site_id').wind_speed.agg(['mean']).to_dict()
for sid in test_df.site_id.unique():
    test_df.loc[test_df.site_id == sid, ['wind_speed']] = test_df.loc[
        test_df.site_id == sid, ['wind_speed']].fillna(mean_df_dict['mean'][sid])
    
    
mean_df_dict = test_df.groupby('site_id').wind_direction.agg(['mean']).to_dict()
for sid in test_df.site_id.unique():
    test_df.loc[test_df.site_id == sid, ['wind_direction']] = test_df.loc[
        test_df.site_id == sid, ['wind_direction']].fillna(mean_df_dict['mean'][sid])
    
mean_df_dict = test_df.groupby('site_id').dew_temperature.agg(['mean']).to_dict()
for sid in test_df.site_id.unique():
    test_df.loc[test_df.site_id == sid, ['dew_temperature']] = test_df.loc[
        test_df.site_id == sid, ['dew_temperature']].fillna(mean_df_dict['mean'][sid])
    
mean_df_dict = test_df.groupby('site_id').air_temperature.agg(['mean']).to_dict()
for sid in test_df.site_id.unique():
    test_df.loc[test_df.site_id == sid, ['air_temperature']] = test_df.loc[
        test_df.site_id == sid, ['air_temperature']].fillna(mean_df_dict['mean'][sid])
mean_df_dict = train_df.groupby('site_id').precip_depth_1_hr.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    if sid == 1 or sid == 5 | sid == 12 :
        continue
    train_df.loc[train_df.site_id == sid, ['precip_depth_1_hr']] = train_df.loc[
        train_df.site_id == sid, ['precip_depth_1_hr']].fillna(mean_df_dict['mean'][sid])
    
    
    
mean_df_dict = test_df.groupby('site_id').precip_depth_1_hr.agg(['mean']).to_dict()
for sid in test_df.site_id.unique():
    if sid == 1 or sid == 5 | sid == 12 :
        continue
    test_df.loc[test_df.site_id == sid, ['precip_depth_1_hr']] = test_df.loc[
        test_df.site_id == sid, ['precip_depth_1_hr']].fillna(mean_df_dict['mean'][sid])
mean_df_dict = train_df.groupby('site_id').sea_level_pressure.agg(['mean']).to_dict()
for sid in train_df.site_id.unique():
    if sid == 5:
        continue
    train_df.loc[train_df.site_id == sid, ['sea_level_pressure']] = train_df.loc[
        train_df.site_id == sid, ['sea_level_pressure']].fillna(mean_df_dict['mean'][sid])
    
mean_df_dict = test_df.groupby('site_id').sea_level_pressure.agg(['mean']).to_dict()
for sid in test_df.site_id.unique():
    if sid == 5:
        continue
    test_df.loc[test_df.site_id == sid, ['sea_level_pressure']] = test_df.loc[
        test_df.site_id == sid, ['sea_level_pressure']].fillna(mean_df_dict['mean'][sid])
train_df.isnull().sum() * 100 / train_df.shape[0]
test_df.isnull().sum() * 100 / test_df.shape[0]
# You can find location of these site_ids and fill cloud_coverage| precip_Depth_1_hr | sea_level_pressure 
# for these locations. but I can't do it right now. so just put -999 for simplicity.
values = {'cloud_coverage': -999, 'precip_depth_1_hr': -999, 'sea_level_pressure': -999}
train_df.fillna(value=values, inplace=True)
test_df.fillna(value=values, inplace=True)
test_df.isnull().sum()
train_df.to_csv('train_filled.csv', index=False)
test_df.to_csv('test_filled.csv', index=False)
