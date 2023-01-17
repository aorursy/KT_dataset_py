%matplotlib inline

import json

import os

import pandas as pd

import seaborn as sns

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection  import train_test_split

import numpy as np

import gc

from scipy.stats import norm # for scientific Computing

from scipy import stats, integrate

import matplotlib.pyplot as plt
#load the dataset

ASHRAE_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

ASHRAE_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

building_meta = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
ASHRAE_train.info()
weather_train.info()
## Function to reduce the DF size

def reduce_memory_usage(df, verbose=True):

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
reduce_memory_usage(building_meta)

reduce_memory_usage(weather_train)

reduce_memory_usage(ASHRAE_train)



reduce_memory_usage(weather_test)

reduce_memory_usage(ASHRAE_test)
ASHRAE_train.describe()
building_meta.describe()
print('number of categories used on "primary_use": ',building_meta['primary_use'].nunique())

print('list of categories used on "primary_use" : ')

print(building_meta['primary_use'].unique())
#print train data columns

print('Columns of the building dataset is', building_meta.columns)

print('Columns of the weather_train dataset is', weather_train.columns)

print('Columns of the train dataset is', ASHRAE_train.columns)
building_meta.isnull().sum()
fig, ax = plt.subplots(figsize=(15,7))

sns.heatmap(building_meta.isnull(), yticklabels=False, cmap='viridis')
print("Percentage of missing values in the building_meta dataset")

building_meta.isna().sum()/len(building_meta)*100
print("Percentage of missing values in the train dataset")

ASHRAE_train.isna().sum()/len(ASHRAE_train)*100
print("Percentage of missing values in the weather_train dataset")

weather_train.isna().sum()/len(weather_train)*100
#merging tables

BuildingTrain = building_meta.merge(ASHRAE_train, left_on='building_id', right_on='building_id' , how='left')

BuildingTest = building_meta.merge(ASHRAE_test, left_on='building_id', right_on='building_id' , how='left')

BuildingTrain.shape, BuildingTest.shape
BuildingTrain.head()
del ASHRAE_test

del ASHRAE_train

del building_meta

gc.collect()
BTW_train=BuildingTrain.merge(weather_train,left_on=['site_id','timestamp'],right_on=['site_id','timestamp'],how='left')

BTW_test = BuildingTest.merge(weather_test,left_on=['site_id','timestamp'],right_on=['site_id','timestamp'],how='left')

BTW_train.shape
del BuildingTest

del BuildingTrain

del weather_test

del weather_train

gc.collect()
BTW_train.columns
BTW_train.head()
print("Percentage of missing values in the BTW_train dataset")

BTW_train.isna().sum()/len(BTW_train)*100
BTW_train.hist('sea_level_pressure')

BTW_train[['sea_level_pressure']].describe()
BTW_train.hist('cloud_coverage')

BTW_train[['cloud_coverage']].describe()
BTW_train.hist('precip_depth_1_hr')

BTW_train[['precip_depth_1_hr']].describe()
def plot_dist(df, column):

    plt.figure(figsize=(18,12))

    ax = sns.distplot(df[column].dropna())

    ax.set_title(column+" Distribution", fontsize=16)

    plt.xlabel(column, fontsize=12)

    #plt.ylabel("distribution", fontsize=12)

    plt.show()
BTW_train[['wind_speed']].describe()

plot_dist(BTW_train,'wind_speed')
BTW_train[['air_temperature']].describe()

plot_dist(BTW_train,'air_temperature')
plot_dist(BTW_train, "sea_level_pressure")
sns.boxplot(x = 'meter', y = 'meter_reading', data = BTW_train)
def outlier_function(df, col_name):

    ''' this function detects first and third quartile and interquartile range for a given column of a dataframe

    then calculates upper and lower limits to determine outliers conservatively

    returns the number of lower and uper limit and number of outliers respectively

    '''

    first_quartile = np.percentile(

        np.array(df[col_name].tolist()), 25)

    third_quartile = np.percentile(

        np.array(df[col_name].tolist()), 75)

    IQR = third_quartile - first_quartile

                      

    upper_limit = third_quartile+(3*IQR)

    lower_limit = first_quartile-(3*IQR)

    outlier_count = 0

                      

    for value in df[col_name].tolist():

        if (value < lower_limit) | (value > upper_limit):

            outlier_count +=1

    return lower_limit, upper_limit, outlier_count
print("{} percent of {} are outliers."

      .format((

              (100 * outlier_function(BTW_train, 'meter_reading')[2])

               / len(BTW_train['meter_reading'])),

              'meter_reading'))
# Distribution of the meter reading in meters without zeros

plt.figure(figsize=(12,10))



#list of different meters

meters = sorted(BTW_train['meter'].unique().tolist()) # [0, 1, 2, 3]



# plot meter_reading distribution for each meter

for meter_type in meters:

    subset = BTW_train[BTW_train['meter'] == meter_type]

    sns.kdeplot(np.log1p(subset["meter_reading"]), 

                label=meter_type, linewidth=2)



# set title, legends and labels

plt.ylabel("Density")

plt.xlabel("Meter_reading")

plt.legend(['electricity', 'chilled water', 'steam', 'hot water'])

plt.title("Density of Logartihm(Meter Reading + 1) Among Different Meters", size=14)
BTW_train.head()
BTW_train.columns
corrmat=BTW_train.corr()

fig,ax=plt.subplots(figsize=(12,10))

sns.heatmap(corrmat,annot=True,annot_kws={'size': 12})
BTW_train = BTW_train.drop(columns=['year_built', 'floor_count', 'wind_direction', 'dew_temperature'])

BTW_test = BTW_test.drop(columns=['year_built', 'floor_count','wind_direction', 'dew_temperature'])
BTW_train ['timestamp'] =  pd.to_datetime(BTW_train['timestamp'])

BTW_test ['timestamp'] =  pd.to_datetime(BTW_test['timestamp'])

BTW_train['Month']=pd.DatetimeIndex(BTW_train['timestamp']).month

BTW_test['Month']=pd.DatetimeIndex(BTW_test['timestamp']).month

BTW_train['Day']=pd.DatetimeIndex(BTW_train['timestamp']).day

BTW_test['Day']=pd.DatetimeIndex(BTW_test['timestamp']).day
BTW_train= BTW_train.groupby(['meter',BTW_train['building_id'],'primary_use',BTW_train['Month'], BTW_train['Day']]).agg({'meter_reading':'sum', 'air_temperature': 'mean', 'wind_speed': 'mean', 'precip_depth_1_hr': 'mean', 'cloud_coverage': 'mean', 'square_feet': 'mean'})

BTW_test_1= BTW_test.groupby(['row_id','meter',BTW_test['building_id'],'primary_use',BTW_test['Month'], BTW_test['Day']]).agg({ 'air_temperature': 'mean', 'wind_speed': 'mean', 'precip_depth_1_hr': 'mean', 'cloud_coverage': 'mean', 'square_feet': 'mean'})
BTW_train.isna().sum()
BTW_train = BTW_train.reset_index()
BTW_train.describe()
BTW_train['wind_speed'] = BTW_train['wind_speed'].astype('float32')

BTW_train['air_temperature'] = BTW_train['air_temperature'].astype('float32')

BTW_train['precip_depth_1_hr'] = BTW_train['precip_depth_1_hr'].astype('float32')

BTW_train['cloud_coverage'] = BTW_train['cloud_coverage'].astype('float32')

BTW_test['wind_speed'] = BTW_test['wind_speed'].astype('float32')

BTW_test['air_temperature'] = BTW_test['air_temperature'].astype('float32')

BTW_test['precip_depth_1_hr'] = BTW_test['precip_depth_1_hr'].astype('float32')

BTW_test['cloud_coverage'] = BTW_test['cloud_coverage'].astype('float32')
BTW_train['precip_depth_1_hr'].fillna(method='ffill', inplace = True)

BTW_train['cloud_coverage'].fillna(method='bfill', inplace = True)



BTW_train['wind_speed'].fillna(BTW_train['wind_speed'].mean(), inplace=True)

BTW_train['air_temperature'].fillna(BTW_train['air_temperature'].mean(), inplace=True)



BTW_test['precip_depth_1_hr'].fillna(method='ffill', inplace = True)

BTW_test['cloud_coverage'].fillna(method='bfill', inplace = True)

BTW_test['precip_depth_1_hr'].fillna(BTW_test['precip_depth_1_hr'].mean(), inplace=True)

BTW_test['cloud_coverage'].fillna(BTW_test['cloud_coverage'].mean(), inplace=True)



BTW_test['wind_speed'].fillna(BTW_test['wind_speed'].mean(), inplace=True)

BTW_test['air_temperature'].fillna(BTW_test['air_temperature'].mean(), inplace=True)

BTW_train.isnull().sum()
BTW_train.shape,BTW_train.dtypes
#visualization of meter reading for a specific building_id

# by Day

building_id = 213

plt.figure(figsize=(14, 8))

ax = sns.lineplot(x="Day", y="meter_reading", hue="meter", data=BTW_train[BTW_train['building_id'] == building_id])

plt.title('Meter readings from building_id {}'.format(building_id))

plt.show()
#visualization of meter reading for a specific building_id

# by Month

building_id = 213

plt.figure(figsize=(14, 8))

ax = sns.lineplot(x="Month", y="meter_reading", hue="meter", data=BTW_train[BTW_train['building_id'] == building_id])

plt.title('Meter readings from building_id {}'.format(building_id))

plt.show()
BTW_train.primary_use.unique()
BTW_train.head()
BTW_encoded = BTW_train[:]

BTW_test_encoded = BTW_test[:]
# label encoding 

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

BTW_encoded["primary_use"] = le.fit_transform(BTW_encoded["primary_use"])

BTW_test_encoded["primary_use"] = le.fit_transform(BTW_test_encoded["primary_use"])
BTW_encoded.columns
X = BTW_encoded[['meter', 'building_id', 'primary_use', 'Month', 'Day','air_temperature', 'wind_speed', 'precip_depth_1_hr', 'cloud_coverage',

       'square_feet']]

y = BTW_encoded['meter_reading']
x_train, x_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state= 45)
from sklearn import preprocessing

from keras import backend as K

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization

from keras.models import Sequential

from keras.optimizers import RMSprop,Adam

from keras import regularizers
def root_mean_squared_error(y_true, y_pred):

  return K.sqrt(K.mean(K.square(y_pred - y_true)))
def make_model(input_dim=10,metrics=root_mean_squared_error,loss='mse', optimizer="rmsprop",drop_rate=0.5):



  model = Sequential()

  model.add(LSTM(128,return_sequences=True, input_shape=(None,input_dim)))

  model.add(Dropout(drop_rate))

  model.add(BatchNormalization())

  model.add(LSTM(128,return_sequences=False))

  model.add(BatchNormalization())

  model.add(Dropout(drop_rate))

  model.add(Dense(1))

  model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

  

  return model
def run_model(model,x_train,y_train,epochs=50,batch_size=500,verbose=1,validation_data=(x_val,y_val),callbacks =None):

  x_train = x_train.values[:]

  x_train= x_train.reshape((x_train.shape[0],1,x_train.shape[-1]))

  y_train = np.log1p(y_train)

  if validation_data != None:

    x_val = validation_data[0].values[:]

    x_val = x_val.reshape((x_val.shape[0],1,x_val.shape[-1]))

    y_val = np.log1p(validation_data[-1])

      

  return model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=verbose,validation_data=(x_val,y_val),callbacks=callbacks)
#best_model_file = "my_model.h5"

#mc = ModelCheckpoint(best_model_file, monitor='val_loss', mode='auto',verbose=True, save_best_only=True)

es = EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, verbose=True, mode='auto')
model = make_model(input_dim=x_train.shape[-1],drop_rate=0.2)
model.summary()
history = run_model(model,x_train,y_train,epochs=30,batch_size=500,verbose=1,validation_data=(x_val,y_val), callbacks =[es]) # callbacks =[mc, es]
loss = history.history

loss.keys()
#rmse loss

rmse_loss_train = loss['root_mean_squared_error']

rmse_loss_val = loss['val_root_mean_squared_error']

epochs_stops = es.stopped_epoch +1 # epochs number from early stopping

epochs = range(1,epochs_stops + 1)  #len(loss_train)

plt.figure(figsize=(12,6))

plt.plot(epochs,rmse_loss_train,'r', label='RMSE train loss')

plt.plot(epochs,rmse_loss_val,'b',label='RMSE val loss')

plt.title(' root mean square error loss')

plt.legend()

plt.show()
# submit = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
# x_test = BTW_test[['meter', 'building_id', 'primary_use', 'Month', 'Day','air_temperature', 'wind_speed', 'precip_depth_1_hr', 'cloud_coverage',#

#        'square_feet']]

# x_test = x_test.values[:]

# x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[-1]))
# prediction = history.predict(x_test)
# prediction = np.expm1(prediction)
# submit['meter_reading'] = prediction
# submit.to_csv('submission.csv', index=False,float_format='%.4f')