import xgboost as xgb

import lightgbm as lgb
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_log_error, mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

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

 

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
build_meta = pd.read_csv('building_metadata.csv')

build_meta = reduce_mem_usage(build_meta)

weather_train = pd.read_csv('weather_train.csv')

weather_train = reduce_mem_usage(weather_train)

weather_test = pd.read_csv('weather_test.csv')

weather_test = reduce_mem_usage(weather_test)

train_df = pd.read_csv('train.csv')

train_df = reduce_mem_usage(train_df)

test_df = pd.read_csv('test.csv')

test_df = reduce_mem_usage(test_df)
print(train_df.shape[0])

print(test_df.shape[0])
gc.collect()
import dask.dataframe as dd
# Join building & weather based on site_id

train_build_df = dd.merge(train_df,build_meta,on='building_id',how='left')

print(train_build_df.shape[0])

train_2_df = dd.merge(train_build_df,weather_train,on=['site_id','timestamp'],how='left')

print(train_2_df.shape[0])



# Join test with building meta data & weather data

test_build_df = dd.merge(test_df,build_meta,on='building_id',how='left')

print(test_build_df.shape[0])

test_2_df = dd.merge(test_build_df,weather_test,on=['site_id','timestamp'],how='left')

print(test_2_df.shape[0])
del train_build_df

del test_build_df
gc.collect()
print(train_2_df.info())
print(test_2_df.info())
print(train_2_df.isna().sum())
print(test_2_df.isna().sum())
train_final_df = train_2_df[['building_id','meter','timestamp','site_id','primary_use','square_feet','meter_reading']]
gc.collect()
train_final_df['timestamp'] = pd.to_datetime(train_final_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
train_final_df.info()
train_final_df.head()
train_final_df['meter'] = train_final_df['meter'].astype('category')
train_final_df['Day_of_Week'] = train_final_df['timestamp'].dt.weekday

train_final_df['Hour_of_Day'] = train_final_df['timestamp'].dt.hour
def Weekday_Weekend(df):

    df['Weekend'] = df['Day_of_Week'].apply(lambda x: 'Y' if x>5 else 'N')



def Office_Hour(df):

    df['OfficeTime'] = df['Hour_of_Day'].apply(lambda x: 'Y' if (x>=7 & x<=18) else 'N')
from dask.multiprocessing import get
ddata_train = dd.from_pandas(train_final_df,npartitions=20)
ddata_train.head()
ddout = ddata_train.map_partitions(Weekday_Weekend)

result = ddout.compute()



ddout = ddata_train.map_partitions(Office_Hour)

result = ddout.compute()

ddata_train.head()
del train_final_df
# COnvert back to pandas dataframe

train_final_1_df = ddata_train.compute()

train_final_1_df.info()
train_final_1_df = train_final_1_df[['building_id','meter','primary_use','square_feet','Weekend','OfficeTime','Hour_of_Day','meter_reading']]

train_final_1_df = pd.get_dummies(train_final_1_df)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
import pickle

# Split the train to 2 data sets, using 1 for training & other for validation

nrow = train_final_1_df.shape[0]

train = train_final_1_df[:round(nrow/2)]

test = train_final_1_df[round(nrow/2)+1:]





# Split to train & validation

#train , test = train_test_split(train_final_1_df, test_size = 0.3)





x_train = train.drop('meter_reading', axis=1)

y_train = train['meter_reading']

x_test = test.drop('meter_reading', axis = 1)

y_test = test['meter_reading']





# Scale the features

#scaler = MinMaxScaler(feature_range=(0, 1)

#x_train_scaled = scaler.fit_transform(x_train)

#x_train = pd.DataFrame(x_train_scaled)

#x_test_scaled = scaler.fit_transform(x_test)

#x_test = pd.DataFrame(x_test_scaled)





from sklearn.model_selection import GridSearchCV

import sklearn

# Set parameters for grid search

param_grid = [

{'n_estimators':[100,150],

'criterion':['mae'],

'max_features':['auto'],

'min_impurity_decrease':[0],

'n_jobs':[-1],

'bootstrap':[False]}

]



# Fit the model

xgbreg = RandomForestRegressor(random_state=123,n_estimators=200,n_jobs=-1,verbose=2)

xgbreg.fit(x_train, y_train)



# Predict for test

#best_model_RF = grid_search.best_estimator_

y_pred=xgbreg.predict(x_test)

print('RMSLE :',np.sqrt(mean_squared_log_error(y_test+1,y_pred+1)))

model_name = 'Model_1.sav'

pickle.dump(xgbreg, open(model_name, 'wb'))







# 2nd round

train = train_final_1_df[round(nrow/2)+1:]

test = train_final_1_df[:round(nrow/2)]





# Split to train & validation

#train , test = train_test_split(train_final_1_df, test_size = 0.3)





x_train = train.drop('meter_reading', axis=1)

y_train = train['meter_reading']

x_test = test.drop('meter_reading', axis = 1)

y_test = test['meter_reading']





# Scale the features

#scaler = MinMaxScaler(feature_range=(0, 1)

#x_train_scaled = scaler.fit_transform(x_train)

#x_train = pd.DataFrame(x_train_scaled)

#x_test_scaled = scaler.fit_transform(x_test)

#x_test = pd.DataFrame(x_test_scaled)





from sklearn.model_selection import GridSearchCV

import sklearn

# Set parameters for grid search

param_grid = [

{'n_estimators':[100,150],

'criterion':['mae'],

'max_features':['auto'],

'min_impurity_decrease':[0],

'n_jobs':[-1],

'bootstrap':[False]}

]



# Fit the model

xgbreg = RandomForestRegressor(random_state=123,n_estimators=200,n_jobs=-1,verbose=2)

xgbreg.fit(x_train, y_train)



# Predict for test

#best_model_RF = grid_search.best_estimator_

y_pred=xgbreg.predict(x_test)

print('RMSLE :',sqrt(metrics.mean_squared_log_error(y_test+1,y_pred+1)))



model_name = 'Model_2.sav'

pickle.dump(xgbreg, open(model_name, 'wb'))
# Predict for test

#best_model_RF = grid_search.best_estimator_

y_pred=xgbreg.predict(x_test)

print('RMSLE :',np.sqrt(mean_squared_log_error(y_test+1,y_pred+1)))



model_name = 'Model_2.sav'

pickle.dump(xgbreg, open(model_name, 'wb'))
gc.collect()
test_final_df = test_2_df[['building_id','meter','timestamp','site_id','primary_use','square_feet']]

test_final_df['timestamp'] = pd.to_datetime(test_final_df['timestamp'], format='%Y-%m-%d %H:%M:%S')

test_final_df['meter'] = test_final_df['meter'].astype('category')

test_final_df['Day_of_Week'] = test_final_df['timestamp'].dt.weekday

test_final_df['Hour_of_Day'] = test_final_df['timestamp'].dt.hour



ddata_test = dd.from_pandas(test_final_df,npartitions=40)

ddout_test = ddata_test.map_partitions(Weekday_Weekend)

result = ddout_test.compute()



ddout_test = ddata_test.map_partitions(Office_Hour)

result = ddout_test.compute()





# COnvert back to pandas dataframe

test_final_1_df = ddata_test.compute()

test_final_1_df.info()



test_final_1_df = test_final_1_df[['building_id','meter','primary_use','square_feet','Weekend','OfficeTime','Hour_of_Day']]

test_final_1_df = pd.get_dummies(test_final_1_df)
gc.collect()
# Load the models 

model_name1 = 'Model_1.sav'

model_name2 = 'Model_2.sav'

model1 = pickle.load(open(model_name1, 'rb'))

model2 = pickle.load(open(model_name2, 'rb'))

y_pred_1 = (model1.predict(test_final_1_df))

y_pred_2 = (model2.predict(test_final_1_df))

result = pd.DataFrame({'row_id':test_2_df['row_id'],'pred_1':y_pred_1,'pred_2':y_pred_2})
gc.collect()

result['meter_reading'] = (result['pred_1']+result['pred_2'])/2

final = result[['row_id','meter_reading']]

final.to_csv('submission.csv',index=False)