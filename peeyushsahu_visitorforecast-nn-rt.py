# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import geohash as gs
import time
import pprint
import datetime
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read all the dataframes into memory and print headers for information

input_folder = '../input' #"../input"

air_store_info = pd.read_csv(os.path.join(input_folder, "air_store_info.csv"))
print('air_store_info')
print(air_store_info.head())
print(air_store_info.shape)
print('#'*10)

air_reserve = pd.read_csv(os.path.join(input_folder, "air_reserve.csv"))
print('air_reserve')
print(air_reserve.head())
print(air_reserve.shape)
print('#'*10)

air_visit_data = pd.read_csv(os.path.join(input_folder, "air_visit_data.csv"))
print('air_visit_data')
print(air_visit_data.head())
print(air_visit_data.shape)
print('#'*10)

store_id_relation = pd.read_csv(os.path.join(input_folder, "store_id_relation.csv"))
print('store_id_relation')
print(store_id_relation.head())
print(store_id_relation.shape)
print('#'*10)

hpg_reserve = pd.read_csv(os.path.join(input_folder, "hpg_reserve.csv"))
print('hpg_reserve')
print(hpg_reserve.head())
print(hpg_reserve.shape)
print('#'*10)

hpg_store_info = pd.read_csv(os.path.join(input_folder, "hpg_store_info.csv"))
print('hpg_store_info')
print(hpg_store_info.head())
print(hpg_store_info.shape)
print('#'*10)

sample_submission = pd.read_csv(os.path.join(input_folder, "sample_submission.csv"))
print('sample_submission')
print(sample_submission.head())
print(sample_submission.shape)
print('#'*10)

date_info = pd.read_csv(os.path.join(input_folder, "date_info.csv"))
date_info = date_info.rename(columns={'calendar_date': 'visit_date'})
print('date_info')
print(date_info.head())
print(date_info.shape)
print('#'*10)

# Return genre from genre_ID
def get_genre_id2name(ID):
    return hotels_genre_ids[ID][0]

#get_genre_id2name(32)

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

def plot_actual_predicted(actual, predicted):
    print('RMSE: ', RMSLE(actual, predicted))
    tmp = pd.DataFrame({'actual': actual, 'predicted': predicted}).sort_values(['actual'])
    plt.scatter(range(tmp.shape[0]), tmp['predicted'], color='green')
    plt.scatter(range(tmp.shape[0]), tmp['actual'], color='blue')
    plt.show()
    del tmp

# Put one more feature: period of the month
## first (1-10 days); second (11-20); third(21-end)
def month_perd(x):
    #print(x.day)
    x = x.day
    if (x < 11):
        return 'first'
    elif (x > 10) & (x < 21):
        return 'second'
    if (x > 20):
        return 'third'
    
    
## Converting latitude and longitude to geohash and applies holiday hack
def geohash_holiday_hack(df):
    # df['geohash'] = 0
    # for ind, row in df.iterrows():
        # df.loc[ind, 'geohash'] = gs.encode(row['latitude'], row['longitude'])

    ## Use the holiday hack
    ### Get all the holiday dates
    hol_un_dates = df.loc[df.holiday_flg==1,:]['visit_date']
    hol_un_dates = hol_un_dates.dt.strftime('%Y-%m-%d').unique()
    #print(hol_un_dates)

    ### Select holiday days
    for date in hol_un_dates:
        # First make holiday as saturday
        df.loc[df.visit_date == date, 'visit_dow'] = 5

        # Find last and next date from holiday
        last_dt = datetime.datetime.strptime('2016-12-23', '%Y-%m-%d') - datetime.timedelta(days=1)
        next_dt = datetime.datetime.strptime('2016-12-23', '%Y-%m-%d') + datetime.timedelta(days=1)

        if (last_dt.weekday() in [0,1,2,3]):
            last_dt = datetime.datetime.strftime(last_dt, '%Y-%m-%d')
            df.loc[df.visit_date == last_dt, 'visit_dow'] = 4
        if (next_dt.weekday() in [1,2,3,4]):
            next_dt = datetime.datetime.strftime(next_dt, '%Y-%m-%d')
            df.loc[df.visit_date == next_dt, 'visit_dow'] = 0
    return df
# Visitor based features
print(air_visit_data.head())
air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'])
air_visit_data["visit_dow"] = air_visit_data.visit_date.dt.dayofweek

tmp1 = air_visit_data.groupby(["air_store_id", "visit_dow"], as_index=False)["visitors"].max().rename(columns={'visitors':'visitors_max'})
tmp2 = air_visit_data.groupby(["air_store_id", "visit_dow"], as_index=False)["visitors"].min().rename(columns={'visitors':'visitors_min'})
tmp3 = air_visit_data.groupby(["air_store_id", "visit_dow"], as_index=False)["visitors"].mean().rename(columns={'visitors':'visitors_mean'})
tmp4 = air_visit_data.groupby(["air_store_id", "visit_dow"], as_index=False)["visitors"].sum().rename(columns={'visitors':'visitors_sum'})
tmp5 = air_visit_data.groupby(["air_store_id", "visit_dow"], as_index=False)["visitors"].median().rename(columns={'visitors':'visitors_median'})
tmp1 = pd.concat([tmp1, tmp2["visitors_min"], tmp3["visitors_mean"], tmp4["visitors_sum"], tmp5["visitors_median"]], axis=1)


store_ids = air_visit_data["air_store_id"].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': store_ids, 'visit_dow': [i]*len(store_ids)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
stores = pd.merge(stores, tmp1, on=["air_store_id", "visit_dow"], how="left")
print(stores.shape)
print(stores.head())
print(stores.isnull().sum())
# Group dataframe based on date and make features for reserve_calls

## Map hpg ids to air ids
print('HPG reservations:', hpg_reserve.shape)
hpg_air_reserve = pd.merge(hpg_reserve, store_id_relation, on='hpg_store_id', how='inner')
hpg_air_reserve = hpg_air_reserve.drop('hpg_store_id', axis=1)
print('HPG with AIR-id  reservations:', hpg_air_reserve.shape)

## Combine hpg and air ids to one dataframe
hpg_air_reserve = pd.concat([air_reserve, hpg_air_reserve], ignore_index=True, axis=0)
print('Combined AIR and HPG reservations:', hpg_air_reserve.shape)
print('Count of unique AIR-ID:', len(hpg_air_reserve.air_store_id.unique()))

## Converting datetime to pandas.datatime
hpg_air_reserve['visit_datetime'] = pd.to_datetime(hpg_air_reserve['visit_datetime'])
hpg_air_reserve['reserve_datetime'] = pd.to_datetime(hpg_air_reserve['reserve_datetime'])
hpg_air_reserve['visit_month'] = hpg_air_reserve.visit_datetime.dt.month
hpg_air_reserve['visit_dow'] = hpg_air_reserve.visit_datetime.dt.dayofweek
hpg_air_reserve['visit_datetime'] = hpg_air_reserve.visit_datetime.dt.date
hpg_air_reserve['reserve_datetime'] = hpg_air_reserve.reserve_datetime.dt.date

## Feature: reservation in advance
hpg_air_reserve['rsv_in_adv'] = (hpg_air_reserve['visit_datetime'] - hpg_air_reserve['reserve_datetime']).dt.days
hpg_air_reserve['rsv_in_adv'] = hpg_air_reserve['rsv_in_adv'].apply(lambda x: x if x >= 7 else 0)
print(hpg_air_reserve.tail())

## Group and reduce rows for unique values 

tmp1 = hpg_air_reserve.groupby(['air_store_id','visit_dow'], as_index=False)['reserve_visitors'].sum()
tmp2 = hpg_air_reserve.groupby(['air_store_id','visit_dow'], as_index=False)['rsv_in_adv'].median().rename(columns={'rsv_in_adv':'res_adv_med'})
tmp3 = hpg_air_reserve.groupby(['air_store_id','visit_dow'], as_index=False)['rsv_in_adv'].sum().rename(columns={'rsv_in_adv':'res_adv_sum'})
tmp4 = hpg_air_reserve.groupby(['air_store_id','visit_dow'], as_index=False)['reserve_visitors'].mean().rename(columns={'reserve_visitors':'res_vis_mean'})
tmp5 = hpg_air_reserve.groupby(['air_store_id','visit_dow'], as_index=False)['reserve_visitors'].max().rename(columns={'reserve_visitors':'res_vis_max'})
tmp6 = hpg_air_reserve.groupby(['air_store_id','visit_dow'], as_index=False)['reserve_visitors'].min().rename(columns={'reserve_visitors':'res_vis_min'})
tmp7 = hpg_air_reserve.groupby(['air_store_id','visit_dow'], as_index=False)['reserve_visitors'].median().rename(columns={'reserve_visitors':'res_vis_median'})



reserve_df = pd.concat([tmp1, tmp2['res_adv_med'], tmp3['res_adv_sum'], tmp4['res_vis_mean'], tmp5['res_vis_max'], tmp6['res_vis_min'], tmp7['res_vis_median']], axis=1) #
#reserve_df['visit_datetime'] = pd.to_datetime(reserve_df['visit_datetime'])
#reserve_df = reserve_df.rename(columns={'visit_datetime':'visit_date'})
#reserve_df = reserve_df.sort_values('visit_datetime', ascending=True)
print('Shape of new reduced df:', reserve_df.shape)
print(reserve_df.dtypes)
print(reserve_df.tail())
print(reserve_df.shape)
reserve_df[reserve_df['air_store_id'] == 'air_fee8dcf4d619598e']
# Training Data
train_data = air_visit_data.copy(deep=True)
train_data['visit_date'] = pd.to_datetime(train_data['visit_date'])
train_data['visit_dow'] = train_data['visit_date'].dt.dayofweek
train_data['visit_month'] = train_data['visit_date'].dt.month
train_data['mont_trimister'] = train_data['visit_date'].apply(month_perd)
print('Shape of training data:', train_data.shape)
print(train_data.head())

# Merging area info and genre
train_data = pd.merge(train_data, air_store_info, on=['air_store_id'], how='left')

# Merging holiday information
date_info['visit_date'] = pd.to_datetime(date_info['visit_date'])
train_data = pd.merge(train_data, date_info, on=['visit_date'], how='left')


# Test Data
test_data = sample_submission.copy(deep=True)
print('Shape of test data:', test_data.shape)
print(test_data.head())
test_data['air_store_id'] = test_data['id'].apply(lambda x: '_'.join(x.split('_')[:2]))
test_data['visit_date'] = test_data['id'].apply(lambda x: x.split('_')[2])
test_data['visit_date'] = pd.to_datetime(test_data['visit_date'])
test_data['visit_dow'] = test_data['visit_date'].dt.dayofweek
test_data['visit_month'] = test_data['visit_date'].dt.month
test_data['mont_trimister'] = test_data['visit_date'].apply(month_perd)
print('Shape of training data:', test_data.shape)
print(test_data.head())

# Merging area info and genre
test_data = pd.merge(test_data, air_store_info, on=['air_store_id'], how='left')

# Merging holiday information
date_info['visit_date'] = pd.to_datetime(date_info['visit_date'])
test_data = pd.merge(test_data, date_info, on=['visit_date'], how='left')



## Merge training data with reduced data from last cell 'reserve_df'
train_data = pd.merge(train_data, stores, on=['air_store_id','visit_dow'], how='left').merge(reserve_df, on=['air_store_id','visit_dow'], how='left')
print('Shape of new training data:', train_data.shape)
print(train_data.head())
print(train_data.isnull().sum())

## Merge test data with reduced data from last cell 'reserve_df'
test_data = pd.merge(test_data, stores, on=['air_store_id','visit_dow'], how='left').merge(reserve_df, on=['air_store_id','visit_dow'], how='left')
print('Shape of new training data:', test_data.shape)
print(test_data.head())
print(test_data.isnull().sum())
# From Surprize me 2
train_data['var_max_lat'] = train_data['latitude'].max() - train_data['latitude']
train_data['var_max_long'] = train_data['longitude'].max() - train_data['longitude']
test_data['var_max_lat'] = test_data['latitude'].max() - test_data['latitude']
test_data['var_max_long'] = test_data['longitude'].max() - test_data['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train_data['lon_plus_lat'] = train_data['longitude'] + train_data['latitude'] 
test_data['lon_plus_lat'] = test_data['longitude'] + test_data['latitude']
# Plotting latitude and longitude
sns.set(style='whitegrid')
plt.figure(figsize=(6,6))
g1 = plt.scatter(train_data.longitude, train_data.latitude, c='b', label='Restaurant_training')
g2 = plt.scatter(test_data.longitude, test_data.latitude, c='r', marker='+', label='Restaurant_test')
plt.legend()
plt.grid()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('This looks like japan :D')
plt.show()
# Adding geohash for coordinates and applying holiday hack
train_data = geohash_holiday_hack(train_data)
test_data = geohash_holiday_hack(test_data)

# Fill NAs with -1
train_data = train_data.fillna(-1)
test_data = test_data.fillna(-1)

print(train_data.head())
print(test_data.head())
# I am a bit confused what to use LabelEncoder or DummyVariable
## We will try both later

print(train_data.columns)
lbl = preprocessing.LabelEncoder() 

lbl1 = lbl.fit(test_data['air_genre_name'].unique())
test_data['air_genre_name'] = lbl1.transform(test_data['air_genre_name'])
train_data['air_genre_name'] = lbl1.transform(train_data['air_genre_name'])
print(lbl1.classes_)

lbl2 = lbl.fit(["first", "second", "third"])
test_data['mont_trimister'] = lbl2.fit_transform(test_data['mont_trimister'])
train_data['mont_trimister'] = lbl2.fit_transform(train_data['mont_trimister'])
print(lbl2.classes_)

lbl3 = lbl.fit(test_data['air_area_name'].unique())
test_data['air_area_name'] = lbl3.transform(test_data['air_area_name'])
train_data['air_area_name'] = lbl3.transform(train_data['air_area_name'])
print(lbl3.classes_)
# First we will convert some features to categorical features
## get_dummie variables
features = pd.get_dummies(air_hpg_com_reduce_df_ml)

# Display the first 5 rows of the last 12 columns
features.head()
# Plotting latitude and longitude
df_plot = train_data
fig, ax = plt.subplots(3,2, figsize=(18,18))
ax[0,0].scatter(np.log2(df_plot.res_adv_med), np.log2(df_plot.visitors))
ax[0,0].set_title('res_adv_med vs visitor')
ax[0,1].scatter(np.log2(df_plot.reserve_visitors), np.log2(df_plot.visitors))
ax[0,1].set_title('reserve_visitors vs visitor')
ax[1,0].scatter(np.log2(df_plot.res_vis_median), np.log2(df_plot.visitors))
ax[1,0].set_title('res_vis_median vs visitor')
ax[1,1].scatter(df_plot.visit_month, np.log2(df_plot.visitors))
ax[1,1].set_title('visit_month vs visitor')
ax[2,0].scatter(df_plot.air_genre_name, np.log2(df_plot.visitors))
ax[2,0].set_title('air_genre_name vs visitor')
ax[2,1].scatter(np.log2(df_plot.res_adv_sum), np.log2(df_plot.visitors))
ax[2,1].set_title('res_adv_sum vs visitor')
print(train_data.columns)
print(test_data.columns)
drop_columns_train = ['air_store_id', 'visit_date', 'day_of_week', 'latitude', 'longitude']
drop_columns_test = ['id', 'air_store_id', 'visit_date', 'visitors', 'day_of_week', 'latitude', 'longitude']

train_data_ml = train_data.drop(drop_columns_train, axis=1)
test_data_ml = test_data.drop(drop_columns_test, axis=1)

train_data_ml['visitors'] = np.log2(train_data_ml['visitors'])
train_data_ml.head()
## Convert data to arrays
# Labels are the values we want to predict
labels = np.array(train_data_ml['visitors'])

# Remove the labels from the features
# axis 1 refers to the columns
features= train_data_ml.drop('visitors', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)
# Creatimg training and test set
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.05, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

from xgboost import XGBRegressor
xgb_model = XGBRegressor(learning_rate=0.2, random_state=2, n_estimators=1000, subsample=0.8, 
                      colsample_bytree=0.75, max_depth=10, reg_lambda=.25, objective='reg:linear', nthread=5)

xgb_model.fit(train_data_ml.drop('visitors', axis=1), train_data_ml['visitors'])
test_xgb_ml = train_data_ml.drop('visitors', axis=1)[:5000]
xgb_preds = xgb_model.predict(test_xgb_ml)
print(xgb_preds[:50])
plot_actual_predicted(train_data_ml.visitors[:5000], xgb_preds)
import xgboost as xgb
xgb.plot_importance(xgb_model)
test_values = pd.DataFrame({'Prediction':xgb_preds,'Truth':train_data_ml.visitors[:5000]})
print(test_values.sort_values('Truth', ascending=True).head(10))
print(test_values.sort_values('Truth', ascending=True).tail(10))
xgb_preds = xgb_model.predict(test_data_ml)
pred_test_csv = pd.DataFrame({'id': list(test_data.id), 'visitors': np.power(xgb_preds, 2)})
pred_test_csv.to_csv('XGB_prediction_3.csv', index=None)
print(pred_test_csv.head())