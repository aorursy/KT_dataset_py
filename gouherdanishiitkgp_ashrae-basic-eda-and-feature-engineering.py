import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os
print(os.listdir('../input/ashrae-energy-prediction'))
# Function to read and do the initial data interpretation

def read_and_interpret_data(filename):

    path = "../input/ashrae-energy-prediction"

    df = pd.read_csv('{0}/{1}'.format(path,filename))

    print("~~~~~~Shape of the data~~~~~~ : ",df.shape)

    print("~~~~~~Columns and their datatype~~~~~~ : ")

    print(df.info())

    print("~~~~~~Quick Look at the data~~~~~~ : ")

    print(df.head())

    print("~~~~~~Description of the data~~~~~~ : ")

    print(df.describe())

    print("~~~~~~NAs present in the data~~~~~~ : ")

    print(df.isna().sum())

    if 'timestamp' in df.columns: 

        df['timestamp'] = pd.to_datetime(df['timestamp'],format = "%Y-%m-%d %H:%M:%S")

        print("~~~~~~Year of the data~~~~~~ :")

        print(df.timestamp.dt.year.unique())

    return df
df_train = read_and_interpret_data('train.csv')
df_train.groupby('building_id')['meter_reading'].agg(['count','min','max','mean','median','std'])

# We can see that the values for building number 1099 are exceptionally high. These can be safely considered as outliers and can be dropped.
df_train.head()
# Remove outliers

df_train = df_train [df_train['building_id'] != 1099 ]

df_train = df_train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
start = df_train['timestamp'].min()

end = df_train['timestamp'].max()
type(start)
def plot_for_date_range(num_buildings,start_building = 0,start_date=start, end_date=end):

    plt.figure(figsize=(18,15), facecolor='white')

    plot_num = 1

    for i in range(start_building,start_building+num_buildings):

        ax = plt.subplot(num_buildings, 1, plot_num)

        data=df_train[df_train.building_id == i].set_index('timestamp').loc[start_date:end_date]

        data.plot(y='meter_reading', ax=ax, label=i, legend=False)

        ax.set_title(f'Building id {i}')

        plot_num +=1

    

    plt.tight_layout()
plot_for_date_range(num_buildings = 10,start_building = 100)
plot_for_date_range(7,0,'2016-07-01', '2016-08-01')
plot_for_date_range(7,80,'2016-07-01', '2016-08-01')
df_train['month'] = df_train['timestamp'].dt.month

df_train['dayofweek'] = df_train['timestamp'].dt.dayofweek

df_train['hourofday'] = df_train['timestamp'].dt.hour
# Function to reduce the DF size

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
df_building_metadata = read_and_interpret_data('building_metadata.csv')
def plot_hist(df,var_name):

    plt.figure(figsize=(17,8))

    plt.hist(df[var_name],bins = 50)

    plt.title(f"Histogram - {var_name}")

    plt.show()
plot_hist(df_building_metadata,'year_built')
plot_hist(df_building_metadata,'floor_count')
# df_building_metadata = df_building_metadata.drop(['year_built','floor_count'],axis = 1)
import random

def fill_building_data(df, col):

    df_notna = df[df[col].notnull()]

    df_na = df[~df[col].notnull()]

    filler_list = df[col].value_counts().index.tolist()[0:5]

    df_na[col] = random.choices(filler_list,k = len(df_na))

    print(df_na.head())

    print(df_na.isna().sum())

    return pd.concat([df_na,df_notna],axis=0).sort_values("building_id")
# df_building_metadata = fill_building_data(df_building_metadata,'year_built')
# df_building_metadata = fill_building_data(df_building_metadata,'floor_count')
df_building_metadata.describe()
df_building_metadata.isna().sum()
df_building_metadata['year_built'].value_counts()
df_building_metadata['year_built'].fillna(1976, inplace = True)
df_building_metadata['floor_count'].fillna(1, inplace = True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_building_metadata["primary_use"] = le.fit_transform(df_building_metadata["primary_use"])
df_building_metadata = reduce_mem_usage(df_building_metadata)
df_weather_train = read_and_interpret_data('weather_train.csv')
for var in ['dew_temperature','air_temperature','wind_speed']:

    plot_hist(df_weather_train,var)
import datetime

def fill_weather_dataset(weather_df):

    

    # Find Missing Dates

    time_format = "%Y-%m-%d %H:%M:%S"

#     start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)

#     end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)

    start_date = weather_df['timestamp'].min().to_pydatetime()

    end_date = weather_df['timestamp'].max().to_pydatetime()

#     total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

#     hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = np.array([(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)])



    missing_hours = []

    for site_id in range(16):

        

        site_tot_hrs = df_weather_train[df_weather_train['site_id'] == 1]['timestamp']

        site_hours = np.array([x.strftime(time_format) for x in site_tot_hrs])

#         site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])

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



    # AIR TEMPERATURE

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

    weather_df.update(air_temperature_filler,overwrite=False)



    # CLOUD COVERAGE

    # Step 1

    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()

    # Step 2

    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler,overwrite=False)



    # DEW TEMPERATURE

    dew_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

    weather_df.update(dew_temperature_filler,overwrite=False)



    # SEA LEVEL PRESSURE

    # Step 1

    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()

    # Step 2

    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])



    weather_df.update(sea_level_filler,overwrite=False)



    # WIND DIRECTION

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

    weather_df.update(wind_direction_filler,overwrite=False)



    # WIND SPEED

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])

    weather_df.update(wind_speed_filler,overwrite=False)



    # PRECIPITATION DEPTH

    # Step 1

    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()

    # Step 2

    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])



    weather_df.update(precip_depth_filler,overwrite=False)



    weather_df = weather_df.reset_index()

    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)

        

    return weather_df



def limit_dew_temp(air_temp, dew_temp):

    if dew_temp > air_temp:

        return air_temp

    else:

        return dew_temp
df_weather_train = fill_weather_dataset(df_weather_train)

df_weather_train['dew_temperature'] = df_weather_train.apply(lambda x: limit_dew_temp(x.air_temperature, x.dew_temperature), axis=1)
# drop_cols = ['cloud_coverage','precip_depth_1_hr','sea_level_pressure','wind_direction']

# df_weather_train = df_weather_train.drop(drop_cols,axis =1)
df_weather_train.head()
df_weather_train.head()
df_weather_train.isna().sum()
# Visualizing distributions after median imputations

for var in ['dew_temperature','air_temperature','wind_speed']:

    plot_hist(df_weather_train,var)
df_weather_train = reduce_mem_usage(df_weather_train)
train_df = pd.merge(df_train,df_building_metadata,on = 'building_id')

# train_df.head()
df_weather_train['timestamp'] = pd.to_datetime(df_weather_train['timestamp'])

train_df = pd.merge(train_df,df_weather_train,on = ['site_id','timestamp'])
train_df['square_feet'] =  np.log1p(train_df['square_feet'])
import gc

target = np.log1p(train_df["meter_reading"])

features = train_df.drop('meter_reading', axis = 1)

del df_train, df_weather_train, train_df

gc.collect()
features=features.drop("timestamp",axis = 1)
features.isna().sum()
features.info()
[var for var in dir() if isinstance(eval(var), pd.core.frame.DataFrame)]
import lightgbm as lgb

from sklearn.model_selection import KFold
categorical_features = ["building_id", "site_id", "meter", "primary_use", "dayofweek","month","hourofday"]

params = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 1280,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse",

}



kf = KFold(n_splits=3)

models = []

for train_index,test_index in kf.split(features):

    train_features = features.loc[train_index]

    train_target = target.loc[train_index]

    

    test_features = features.loc[test_index]

    test_target = target.loc[test_index]

    

    d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_features, free_raw_data=False)

    d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_features, free_raw_data=False)

    

    model = lgb.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=50)

    models.append(model)

    del train_features, train_target, test_features, test_target, d_training, d_test

    gc.collect()
del features, target

gc.collect()
for model in models:

    lgb.plot_importance(model)

    plt.show()
#Import the regression tree model

# from sklearn.tree import DecisionTreeRegressor

# regression_model = DecisionTreeRegressor(criterion="mse",min_samples_leaf=5)
#Fit the model

# x_train = train_df.drop(['meter_reading','timestamp'],axis = 1)

# y_train = train_df['meter_reading']

# regression_model.fit(x_train,y_train)
# x_train.isna().sum()
#Predict on Training Data

# predicted_train = regression_model.predict(x_train)
#Compute the RMSLE

# def RMSLE(pred,act): 

#     return np.sqrt(np.sum((np.log(pred+1)-np.log(act+1))**2)/len(act))
# Training Error

# RMSLE(predicted_train,y_train)
# Checking Actual and Predicted values side by side

# pd.DataFrame(zip(y_train,predicted_train),columns = ['Actual','Predicted']).iloc[2000000:10000000,].head(10)
df_test = read_and_interpret_data('test.csv')
df_test = reduce_mem_usage(df_test)
df_test['month'] = df_test['timestamp'].dt.month

df_test['dayofweek'] = df_test['timestamp'].dt.dayofweek

df_test['hourofday'] = df_test['timestamp'].dt.hour
# df_test.head()
# df_test.info()
df_test = reduce_mem_usage(df_test)
df_weather_test = read_and_interpret_data('weather_test.csv')
for var in ['dew_temperature','air_temperature','wind_speed']:

    plot_hist(df_weather_test,var)
# drop_cols = ['cloud_coverage','precip_depth_1_hr','sea_level_pressure','wind_direction']

# df_weather_test = df_weather_test.drop(drop_cols,axis =1)
df_weather_test.head()
# df_weather_test.isna().sum()
# df_weather_test.fillna(df_weather_test.median(),inplace=True)
# df_weather_test.head()
df_weather_test.isna().sum()
# Visualizing distributions after median imputations

# for var in ['dew_temperature','air_temperature','wind_speed']:

#     plot_hist(df_weather_test,var)
test_df = pd.merge(df_test,df_building_metadata,on = 'building_id')



test_df = pd.merge(test_df,df_weather_test,on = ['site_id','timestamp'],how='left')
test_df = reduce_mem_usage(test_df)
[var for var in dir() if isinstance(eval(var), pd.core.frame.DataFrame)]
del df_test, df_weather_test, df_building_metadata

gc.collect()
row_ids = test_df['row_id']

test_df = test_df.drop(['timestamp','row_id'],axis = 1)
test_df["square_feet"] = np.log1p(test_df["square_feet"])
test_df.fillna(test_df.median(),inplace=True)
test_df.head()
test_df.info()
test_df.isna().sum()
gc.collect()
results = []

for model in models:

    if  results == []:

        results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)

    else:

        results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)

    del model

    gc.collect()
del test_df, models

gc.collect()
#Predict on Testing Data

# predicted_test = regression_model.predict(test_df)
# submission_df = pd.DataFrame(zip(row_id,predicted_test),columns = ['row_id','meter_reading'])

results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(results, 0, a_max=None)})

del row_ids,results

gc.collect()

results_df.to_csv("submission_lgbm1.csv", index=False)
# submission_df = submission_df.sort_values(by = 'row_id')
# submission_df.head()
# submission_df.shape
# submission_df.to_csv("ashrae_submit.csv", index=False)
# from IPython.display import FileLink

# FileLink(r'ashrae_submit.csv')
# df_building_metadata['year_built'].isna().sum()
# plt.figure(figsize=(17,8))

# plt.hist(df_building_metadata['year_built'],bins = 20)

# plt.title("Histogram showing the Distribution of the Year in which Builidings were built")

# plt.show()
# Making a list of mode years

# mode_yr1,mode_yr2 = list(range(1960,1975)),  list(range(2000,2010))

# mode_years = mode_yr1 + mode_yr2
# import random

# random.seed(123)
# Replacing NAs by chooosing randomly from the mode years

# nans = df_building_metadata['year_built'].isna()

# length = sum(nans)

# replacement = random.choices(mode_years, k=length)

# df_building_metadata.loc[nans,'year_built'] = replacement
# plt.figure(figsize=(17,8))

# plt.hist(df_building_metadata['year_built'],bins = 20)

# plt.title("Histogram showing the Distribution of the Year in which Builidings were built")

# plt.show()
# plt.figure(figsize=(17,8))

# plt.hist(df_building_metadata['year_built'],bins = 20)

# plt.title("Histogram showing the Distribution of the Year in which Builidings were built")

# plt.show()
# df_building_metadata[df_building_metadata.year_built.isnull()]
# df_building_metadata['floor_count'].isna().sum()
# plt.figure(figsize=(17,8))

# plt.hist(df_building_metadata['floor_count'],bins = 20)

# plt.title("Histogram showing the distribution of number of Floors in Buildings")

# plt.show()
# Making a list of mode years

# mode_floors = range(10)
# Replacing NAs by chooosing randomly from the mode years

# nans = df_building_metadata['floor_count'].isna()

# length = sum(nans)

# replacement = random.choices(mode_floors, k=length)

# df_building_metadata.loc[nans,'floor_count'] = replacement
# plt.figure(figsize=(17,8))

# plt.hist(df_building_metadata['floor_count'],bins = 20)

# plt.title("Histogram showing the distribution of number of Floors in Buildings")

# plt.show()
# [1,0,np.nan,2].isnull().replace(0)
# nan_yrs = sum(df_building_metadata.year_built.isnull())

# rand_mode_year = random.choices(mode_years, k =nan_yrs)
# df_building_metadata['year_built'].replace(np.nan,rand_mode_year)
# list(range(1960,1975))
# mode_years[0]
# df_building_metadata['year_built'] = df_building_metadata['year_built'].astype('int16')
#df_train['timestamp'].dt.weekday_name.unique()
#df_train['day_of_weak'] = df_train['timestamp'].dt.day
#df_train.building_id.nunique() * 24 * 365 
#df_test = read_and_interpret_data('test.csv')
# df_weather_train = read_and_interpret_data('weather_train.csv')
#df_weather_test = read_and_interpret_data('weather_test.csv')
#df_meta = read_and_interpret_data('building_metadata.csv')
# Reducing dataframe size

# df_train = reduce_mem_usage(df_train)

# df_test = reduce_mem_usage(df_test)

# df_weather_train = reduce_mem_usage(df_weather_train)

# df_weather_test = reduce_mem_usage(df_weather_test)

# df_meta = reduce_mem_usage(df_meta)
# train_df = df_train.join(df_meta.set_index('building_id'), on = 'building_id')
# train_df.info()
# train_df = train_df.join(df_weather_train.set_index('site_id'), on = 'site_id')
# pr = [0 for x in df_test['row_id'] == df_test.index.tolist()]
# len(pr)
# df_test = df_test.drop('row_id',axis=1)
# df_test['meter_reading'] = 'NA'
# df_test.head()
# df_test_train = pd.concat([df_train,df_test])
# del df_train, df_test
# df_test_train[df_test_train.meter_reading == 'NA'].head()
# df_test_train_meta = df_test_train.join(df_meta.set_index('building_id'), on = 'building_id')

# df_test_train_meta.head()