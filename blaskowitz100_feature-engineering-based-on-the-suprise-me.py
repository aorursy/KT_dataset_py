import glob
import re
import numpy as np
import pandas as pd

# Remove the restriction for the max dataframe width
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250

from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from IPython.display import display
data = {
    'gt_visits': pd.read_csv('../input/air_visit_data.csv'),
    'air_store_info': pd.read_csv('../input/air_store_info.csv'),
    #'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'air_reserve_history': pd.read_csv('../input/air_reserve.csv'),
    'hpg_reserve_history': pd.read_csv('../input/hpg_reserve.csv'),
    
    'hpg_to_air_id': pd.read_csv('../input/store_id_relation.csv'),
    'subm_visits': pd.read_csv('../input/sample_submission.csv'),
    'holidays': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
}
###############################################################################################
# Get the air-reserve id  of the hpg restaurants
###############################################################################################
data['hpg_reserve_history'] = pd.merge(
    data['hpg_reserve_history'], data['hpg_to_air_id'], 
    how='inner', on=['hpg_store_id']
)

###############################################################################################
# Drop the HPG id
###############################################################################################
data['hpg_reserve_history'] = data['hpg_reserve_history'].drop('hpg_store_id', axis=1)

display(data['hpg_reserve_history'].shape)
###############################################################################################
# Append the HPG reservations to the air-reserve history
###############################################################################################
print("Shape before: ", data['air_reserve_history'].shape)

reservation_history = data['air_reserve_history'].append(data['hpg_reserve_history'], sort="True")
reservation_history = data['air_reserve_history'].sort_values(by=['air_store_id', 'reserve_datetime'])
reservation_history = data['air_reserve_history'].reset_index()
reservation_history = reservation_history.drop('index', axis=1)

display(reservation_history.head())
print("Shape after: ", reservation_history.shape)
###############################################################################################
# Log transform the the ammount of reserved visitors for this day
###############################################################################################
reservation_history['reserve_visitors'] = np.log1p(reservation_history['reserve_visitors'])
###############################################################################################
# Convert dates into datetime objects / get the day of the week / 
# cut off hours, minutes and seconds
###############################################################################################
reservation_history['visit_datetime'] = pd.to_datetime(reservation_history['visit_datetime'])
reservation_history['visit_dow'] = reservation_history['visit_datetime'].dt.dayofweek
reservation_history['visit_datetime'] = reservation_history['visit_datetime'].dt.date
reservation_history['reserve_datetime'] = pd.to_datetime(reservation_history['reserve_datetime'])
reservation_history['reserve_datetime'] = reservation_history['reserve_datetime'].dt.date

###############################################################################################
# Calculate the time difference between reservation and visit
###############################################################################################
reservation_history['reserve_datetime_diff'] = reservation_history.apply(
    lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
    axis=1
)
reservation_history[reservation_history.air_store_id == 'air_00a91d42b08b08d9']
###############################################################################################
# Group the reservations in to subgroubs:
# 
# EARLY RESERVATIONS
# sum_res_diff_er ==> SUM reservation_diff on this day
# sum_vis_er ==> SUM reservated visitors this day
# avg_res_diff_er ==> AVG reservation_diff on this day
# avg_vis_er ==> AVG reservated visitors this day
#
# LATE RESERVATIONS
# sum_res_diff_lr ==> SUM reservation_diff on this day
# sum_vis_lr ==> SUM reservated visitors this day
# avg_res_diff_lr ==> AVG reservation_diff on this day
# avg_vis_lr ==> AVG reservated visitors this day
###############################################################################################
reservation_history['early_reservation'] = reservation_history['reserve_datetime_diff'] > 2
reservation_history['late_reservation'] = reservation_history['reserve_datetime_diff'] <= 2

# SUM early reservations
tmp1 = reservation_history[reservation_history['early_reservation']].groupby(
    ['air_store_id','visit_datetime'], as_index=False
)[['reserve_datetime_diff', 'reserve_visitors']]
tmp1 = tmp1.sum()
tmp1 = tmp1.rename(columns={
    'visit_datetime':'visit_date',
    'reserve_datetime_diff': 'sum_res_diff_er',
    'reserve_visitors':'sum_vis_er'
})

# AVG early reservations
tmp2 = reservation_history[reservation_history['early_reservation']].groupby(
    ['air_store_id','visit_datetime'], as_index=False
)[['reserve_datetime_diff', 'reserve_visitors']]
tmp2 = tmp2.mean()
tmp2 = tmp2.rename(columns={
    'visit_datetime':'visit_date',
    'reserve_datetime_diff': 'avg_res_diff_er',
    'reserve_visitors':'avg_vis_er'
})

# SUM late reservations
tmp3 = reservation_history[reservation_history['late_reservation']].groupby(
    ['air_store_id','visit_datetime'], as_index=False
)[['reserve_datetime_diff', 'reserve_visitors']]
tmp3 = tmp3.sum()
tmp3 = tmp3.rename(columns={
    'visit_datetime':'visit_date', 
    'reserve_datetime_diff': 'sum_res_diff_lr', 
    'reserve_visitors':'sum_vis_lr'
})

# AVG late reservations
tmp4 = reservation_history[reservation_history['late_reservation']].groupby(
    ['air_store_id','visit_datetime'], as_index=False
)[['reserve_datetime_diff', 'reserve_visitors']]
tmp4 = tmp4.mean()
tmp4 = tmp4.rename(columns={
    'visit_datetime':'visit_date', 
    'reserve_datetime_diff': 'avg_res_diff_lr',
    'reserve_visitors':'avg_vis_lr'
})

reservation_history = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])
reservation_history = pd.merge(reservation_history, tmp3, how='outer', on=['air_store_id','visit_date'])
reservation_history = pd.merge(reservation_history, tmp4, how='outer', on=['air_store_id','visit_date'])

reservation_history.head()
###############################################################################################
# Get all unique stores from the submission file
# Because the submission file contains the restaurant id and visit date in one attribute, 
# this information has to be splitted up
###############################################################################################

data['subm_visits']['visit_date'] = data['subm_visits']['id'].map(lambda x: str(x).split('_')[2])
data['subm_visits']['air_store_id'] = data['subm_visits']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

# Extract unique store ids and create an empty dataframe for the store meta information
unique_stores = data['subm_visits']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

stores.head()
###############################################################################################
# Fill the gaps in the training dataset for each restaurant
# So every row step is exactly one day
###############################################################################################
data['gt_visits']['visit_date'] = pd.to_datetime(data['gt_visits']['visit_date'])
data['gt_visits'] = data['gt_visits'].groupby('air_store_id').resample('D', on='visit_date').sum().fillna(0)
data['gt_visits'] = data['gt_visits'].reset_index()
data['gt_visits'].head()
###############################################################################################
# Also check if the submission data has the same stepsize for each restaurant
# One row step == one day
###############################################################################################
org_shape = data['subm_visits'].shape
data['subm_visits']['visit_date'] = pd.to_datetime(data['subm_visits']['visit_date'])
tmp = data['subm_visits'].groupby('air_store_id').resample('D', on='visit_date').sum().fillna(0)
tmp = tmp.reset_index()
resampled_shape = tmp.shape

if org_shape[0] == resampled_shape[0]:
    print('Submission has a stepsize of one day per row and restaurant')
    del org_shape, tmp, resampled_shape
###############################################################################################
# Transform to datetime objects and split the dates up
###############################################################################################
data['gt_visits']['visit_date'] = pd.to_datetime(data['gt_visits']['visit_date'])
data['gt_visits']['dow'] = data['gt_visits']['visit_date'].dt.dayofweek
data['gt_visits']['year'] = data['gt_visits']['visit_date'].dt.year
data['gt_visits']['month'] = data['gt_visits']['visit_date'].dt.month
data['gt_visits']['week'] = data['gt_visits']['visit_date'].dt.week
data['gt_visits']['visit_date'] = data['gt_visits']['visit_date'].dt.date

# Also store the visit date as an integer value
data['gt_visits']['visit_date_int'] = data['gt_visits']['visit_date'].apply(
    lambda x: x.strftime('%Y%m%d')
).astype(int)

# Also log-transform the ground truth visitor values
data['gt_visits']['visitors'] = np.log1p(data['gt_visits']['visitors'].values.astype(np.int))

data['gt_visits'].head()
###############################################################################################
# Transform to datetime objects and split the dates up
###############################################################################################
data['subm_visits']['visit_date'] = pd.to_datetime(data['subm_visits']['visit_date'])
data['subm_visits']['dow'] = data['subm_visits']['visit_date'].dt.dayofweek
data['subm_visits']['year'] = data['subm_visits']['visit_date'].dt.year
data['subm_visits']['month'] = data['subm_visits']['visit_date'].dt.month
data['subm_visits']['week'] = data['subm_visits']['visit_date'].dt.week
data['subm_visits']['visit_date'] = data['subm_visits']['visit_date'].dt.date

# Also store the visit date as an integer value
data['subm_visits']['visit_date_int'] = data['subm_visits']['visit_date'].apply(
    lambda x: x.strftime('%Y%m%d')
).astype(int)

data['subm_visits'].head()
###############################################################################################
# Calculate the min, max, avg, median and overall reservations for each day of a week
###############################################################################################
# Min visits
tmp = data['gt_visits'].groupby(['air_store_id','dow'], as_index=False)['visitors']
tmp = tmp.min()
tmp = tmp.rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

# Mean visits
tmp = data['gt_visits'].groupby(['air_store_id','dow'], as_index=False)['visitors']
tmp = tmp.mean()
tmp = tmp.rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

# Median visits
tmp = data['gt_visits'].groupby(['air_store_id','dow'], as_index=False)['visitors']
tmp = tmp.median()
tmp = tmp.rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

# Max visits
tmp = data['gt_visits'].groupby(['air_store_id','dow'], as_index=False)['visitors']
tmp = tmp.max()
tmp = tmp.rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

# Overall visits on this week day
tmp = data['gt_visits'].groupby(['air_store_id','dow'], as_index=False)['visitors']
tmp = tmp.count()
tmp = tmp.rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

###############################################################################################
# Merge this information with the remaining restaurant meta information
###############################################################################################
stores = pd.merge(stores, data['air_store_info'], how='left', on=['air_store_id'])

###############################################################################################
# Show one example
###############################################################################################
stores.loc[stores['air_store_id'] == 'air_00a91d42b08b08d9']
###############################################################################################
# Remove some char from the Genre name and area name
###############################################################################################
stores['air_genre_name'] = stores['air_genre_name'].map(
    lambda x: str(str(x).replace('/',' '))
)
stores['air_area_name'] = stores['air_area_name'].map(
    lambda x: str(str(x).replace('-',' '))
)

###############################################################################################
# Label-Encoding the cleanded names
###############################################################################################
le = LabelEncoder()
stores['air_genre_name'] = le.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = le.fit_transform(stores['air_area_name'])
stores['lon_plus_lat'] = stores['longitude'] + stores['latitude']
stores['var_max_lat'] = stores['latitude'].max() - stores['latitude']
stores['var_max_lon'] = stores['longitude'].max() - stores['longitude']
le = LabelEncoder()
stores['air_store_id_feat'] = le.fit_transform(stores['air_store_id'])
display(stores.head())
###############################################################################################
# Prepare the datetime object and simplify it in a da and day of week
###############################################################################################
data['holidays']['visit_date'] = pd.to_datetime(data['holidays']['visit_date'])

# Attention: This day of week does not match the encoding of the 'dow' field
data['holidays']['day_of_week'] = le.fit_transform(data['holidays']['day_of_week'])
data['holidays']['dow_holidays'] = data['holidays']['visit_date'].dt.dayofweek

data['holidays']['visit_date'] = data['holidays']['visit_date'].dt.date

display(data['holidays'].head())
###############################################################################################
# Add holiday information to the training data
###############################################################################################
train = pd.merge(data['gt_visits'], data['holidays'], how='left', on=['visit_date'])
display(train.head())
###############################################################################################
# Add holiday information to the submission data
###############################################################################################
test = pd.merge(data['subm_visits'], data['holidays'], how='left', on=['visit_date'])
display(test.head())
##############################################################################################
# Merge the training data with the prepared meta information
##############################################################################################
train = pd.merge(train, stores, how='inner', on=['air_store_id','dow'])
train = pd.merge(train, reservation_history, how='left', on=['air_store_id','visit_date'])

# Create the same ID that is used in the submission file
train['id'] = train.apply(
    lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), 
    axis=1
)

###############################################################################################
# Merge the submission dataset with the prepared meta information
###############################################################################################
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])
test = pd.merge(test, reservation_history, how='left', on=['air_store_id','visit_date'])
###############################################################################################
# Sort the train and test dataframes again
###############################################################################################
train = train.sort_values(by=['air_store_id', 'visit_date'])
test = test.sort_values(by=['air_store_id', 'visit_date'])
###############################################################################################
# Fill NaN with an -1 value
###############################################################################################
train = train.fillna(-1)
test = test.fillna(-1)
display(train.head())
display(train.shape)
display(test.head())
display(test.shape)
FEATURES = {
    'air_store_id_feat' : 'LabelEncoded store ID as an input feature. It allows the network to seperate between the stores',
    'dow' : 'Day of the week e.g. Monday, Tuesday, Wednesday,...',
    'year' : 'Year of the visit',
    'month' : 'Month of the visit',
    'week' : 'Week of the visit',
    'visit_date_int' : 'Complete visit date as a integer value',
    'holiday_flg' : 'Is the current day in the holidays',
    'min_visitors' : 'Minimum visitors of the current week day',
    'mean_visitors' : 'Mean visitors of the current week day',
    'median_visitors' : 'Median visitors of the current week day',
    'max_visitors' : 'Maximum visitors of the current week day',
    'count_observations' : 'Total number of reservations for this week day',
    'air_genre_name' : 'Label encoded name of the cusine genre',
    'air_area_name' : 'Label encoded name of the area the restaurant is located in',
    'latitude' : 'Latitude of the restaurant location',
    'longitude' : 'Longitude of the restaurant location',
    'lon_plus_lat' : 'Linear combination of longitude and latitude',
    'var_max_lat' : 'Max(Latitude) - Latitude of the current restaurant',
    'var_max_lon' : 'Max(Longitude) - Longitude of the current restaurant',
    'sum_res_diff_er' : 'Summed up differences between the reservation date and the visit date [Diff > 2 days]',
    'sum_vis_er' : 'Summed up reservated visitors for this day [Diff > 2 days]',
    'avg_res_diff_er' : 'AVG of differences between the reservation date and the visit date [Diff > 2 days]',
    'avg_vis_er' : 'AVG reservated visitors for this day [Diff > 2 days]',
    'sum_res_diff_lr' : 'Summed up differences between the reservation date and the visit date [Diff <= 2 days]',
    'sum_vis_lr' : 'Summed up reservated visitors for this day [Diff <= 2 days]',
    'avg_res_diff_lr' : 'AVG of differences between the reservation date and the visit date [Diff > 2 days]',
    'avg_vis_lr' : 'AVG reservated visitors for this day [Diff <= 2 days]' 
}

EXCLUDED_FEATURES = {
    'id' : 'Air reservation id of the restaurant',
    'visit_date' : 'Use the numeric value instead!',
    'air_store_id' : 'Air reservation id of the restaurant',
    'day_of_week' : 'Day of the week encoded in the date_info.csv file',
    'dow_holidays' : 'Day of the week encoded in the date_info.csv file'
}

GROUND_TRUTH_FEATURES = {
    'visitors' : 'Ground truth information. The number os visitors is transformed with np.log1p()'
}
FEATURE_COLS = list(FEATURES.keys())
EXCLUDED_COLS = list(EXCLUDED_FEATURES.keys())
GROUND_TRUTH_COLS = list(GROUND_TRUTH_FEATURES.keys())

print('Number of cols: ', len(FEATURE_COLS) + len(EXCLUDED_COLS) + len(GROUND_TRUTH_COLS))