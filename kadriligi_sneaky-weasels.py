# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# matplotlib and seaborn for plotting

#import matplotlib.pyplot as plt



#import seaborn as sns

#import matplotlib.patches as patches

#import plotly

#from plotly import tools, subplots

#import plotly.offline as py

#py.init_notebook_mode(connected=True)

#import plotly.graph_objs as go

#import plotly.express as px

pd.set_option('max_columns', 150)



#py.init_notebook_mode(connected=True)

#from plotly.offline import init_notebook_mode, iplot

#init_notebook_mode(connected=True)

#import plotly.graph_objs as go



import os

import random

import math

import psutil

import pickle



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}

weather_dtype = {"site_id":"uint8",'air_temperature':"float16",'cloud_coverage':"float16",'dew_temperature':"float16",'precip_depth_1_hr':"float16",

                 'sea_level_pressure':"float32",'wind_direction':"float16",'wind_speed':"float16"}

train_dtype = {'meter':"uint8",'building_id':'uint16'}
%%time



weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv", parse_dates=['timestamp'], dtype=weather_dtype)





metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)



train = pd.read_csv("../input/ashrae-energy-prediction/train.csv", parse_dates=['timestamp'], dtype=train_dtype, nrows = 10000000)

#test = pd.read_csv("../input/ashrae-energy-prediction/test.csv", parse_dates=['timestamp'], usecols=['building_id','meter','timestamp'], dtype=train_dtype)

#weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=['timestamp'], dtype=weather_dtype)

print('Size of train_df data', train.shape)

print('Size of weather_train_df data', weather_train.shape)

#print('Size of weather_test_df data', weather_test.shape)

print('Size of building_meta_df data', metadata.shape)
del train_dtype

del metadata_dtype

del weather_dtype
train.head()
weather_train.head()
metadata.head()
metadata.drop('floor_count',axis=1,inplace=True)
train['Month'] = train['timestamp'].dt.month.astype("uint8")

train['DayOfMonth'] = train['timestamp'].dt.day.astype("uint8")

train['DayOfWeek'] = train['timestamp'].dt.dayofweek.astype("uint8")

train['Hour'] = train['timestamp'].dt.hour.astype("uint8")
%%time

train = pd.merge(train,metadata,on='building_id',how='left')

#test  = pd.merge(test,metadata,on='building_id',how='left')

print ("Training Data+Metadata Shape {}".format(train.shape))

#print ("Testing Data+Metadata Shape {}".format(test.shape))

gc.collect()

train = pd.merge(train,weather_train,on=['site_id','timestamp'],how='left')

#test  = pd.merge(test,weather_test,on=['site_id','timestamp'],how='left')

print ("Training Data+Metadata+Weather Shape {}".format(train.shape))

#print ("Testing Data+Metadata+Weather Shape {}".format(test.shape))

gc.collect()
train.head()
le = LabelEncoder()



train['primary_use']= le.fit_transform(train['primary_use']).astype("uint8")

#test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")

print (train.shape)#, test.shape)

gc.collect()
del le
train.head()
# Drop nonsense entries

# As per the discussion in the following thread, https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083, there is some discrepancy in the meter_readings for different ste_id's and buildings. It makes sense to delete them

idx_to_drop = list((train[(train['site_id'] == 0) & (train['timestamp'] < "2016-05-21 00:00:00")]).index)

print (len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)



# dropping all the electricity meter readings that are 0, after considering them as anomalies.

idx_to_drop = list(train[(train['meter'] == 0) & (train['meter_reading'] == 0)].index)

print(len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)



gc.collect()
del idx_to_drop
train.drop('timestamp',axis=1,inplace=True)

#test.drop('timestamp',axis=1,inplace=True)

train.shape
'''

# Fill NA

cols = ['year_built', 'air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

for col in cols:

    train[col].fillna(np.nanmean(train[col].tolist()),inplace=True)

    #test[col].fillna(np.nanmean(test[col].tolist()),inplace=True)

    '''
cols = train.columns
%%time

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

imp_mean = IterativeImputer(random_state=0)

imp_mean.fit(train)

imputed_data = pd.DataFrame(data=imp_mean.transform(train), columns = cols)



gc.collect()
#del train

del metadata

del weather_train

del cols
imputed_data.head()
y = imputed_data['meter_reading']

imputed_data.drop('meter_reading',axis=1,inplace=True)





%%time

imputed_data['building_id'] = imputed_data['building_id'].astype("uint16")

imputed_data['site_id'] = imputed_data['site_id'].astype("uint8") 

imputed_data['square_feet'] = imputed_data['square_feet'].astype("uint32") 

imputed_data['year_built'] = imputed_data['year_built'].astype("uint16")

imputed_data['sea_level_pressure'] = imputed_data['sea_level_pressure'].astype("uint16")

imputed_data['wind_speed'] = imputed_data['wind_speed'].astype("uint8")

imputed_data['cloud_coverage'] = imputed_data['cloud_coverage'].astype("uint8")

imputed_data['Month'] = imputed_data['Month'].astype("uint8")

imputed_data['meter'] = imputed_data['meter'].astype("uint8")

imputed_data['Hour'] = imputed_data['Hour'].astype("uint8")

    

imputed_data['primary_use'] = imputed_data['primary_use'].astype("uint8")

imputed_data['DayOfWeek'] = imputed_data['DayOfWeek'].astype("uint8")

imputed_data['DayOfMonth'] = imputed_data['DayOfMonth'].astype("uint8")

imputed_data['wind_direction'] = imputed_data['wind_direction'].astype("uint16") 

gc.collect()

imputed_data.dtypes
'''from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=100)

clf = clf.fit(train, y)

clf.feature_importances_  



model = SelectFromModel(clf, prefit=True)

train_new = model.transform(train)

train_new.shape               



gc.collect()

'''
#test_new = model.transform(test)

#test_new.shape
#X_train, X_test, y_train, y_test = train_test_split(train, y, train_size=0.7, test_size=0.3, random_state=0, shuffle=True)

#X_train.shape, X_test.shape
X_train, X_test, y_train, y_test = train_test_split(imputed_data, y, 

                                                    train_size=0.7, test_size=0.3, 

                                                    random_state=0)



gc.collect()
X_train.shape
X_test.shape
categorical_cols = ['building_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth']#

train = pd.get_dummies(X_train, columns=categorical_cols, sparse=True)



test = pd.get_dummies(X_test, columns=categorical_cols, sparse=True)



gc.collect()



train.shape
test.shape
%%time

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error

rf = RandomForestRegressor()

rf.fit(train, y_train)

preds = rf.predict(test)

rmsle = np.sqrt(mean_squared_log_error( y_test, preds))

print(rmsle)
ser = pd.DataFrame(rf.feature_importances_,train.columns,columns=['Importance']).sort_values(by='Importance')

ser[ser['Importance']>0.0001]
del ser
'''

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import mean_squared_error

parameters = {'n_estimators':[50,75,100, 125,150], 

              'max_features':['auto',25, 50, 75, 100], 

              'max_depth': [None, 2,4,6,8]}

rf = RandomForestRegressor()

clf = GridSearchCV(rf, parameters)

clf.fit(train, y_train)



#sorted(clf.cv_results_.keys())'''
train_columns = train.columns
#from sklearn.metrics import mean_squared_error

#preds = clf.predict(X_test)

#mse = mean_squared_error(y_test, preds)

#print(mse)
#clf.best_params_

#clf.best_score_

#clf.n_splits_
del test

del train

del X_train

del X_test

metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}

weather_dtype = {"site_id":"uint8",'air_temperature':"float16",'cloud_coverage':"float16",'dew_temperature':"float16",'precip_depth_1_hr':"float16",

                 'sea_level_pressure':"float32",'wind_direction':"float16",'wind_speed':"float16"}

train_dtype = {'meter':"uint8",'building_id':'uint16'}
%%time

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv", parse_dates=['timestamp'], usecols=['building_id','meter','timestamp'], dtype=train_dtype)

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=['timestamp'], dtype=weather_dtype)

metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)

metadata.drop('floor_count',axis=1,inplace=True)

test.head()
del metadata_dtype

del weather_dtype

del train_dtype
test['Month'] = test['timestamp'].dt.month.astype("uint8")

test['DayOfMonth'] = test['timestamp'].dt.day.astype("uint8")

test['DayOfWeek'] = test['timestamp'].dt.dayofweek.astype("uint8")

test['Hour'] = test['timestamp'].dt.hour.astype("uint8")

gc.collect()
%%time

test  = pd.merge(test,metadata,on='building_id',how='left')

print ("Testing Data+Metadata Shape {}".format(test.shape))

gc.collect()

test  = pd.merge(test,weather_test,on=['site_id','timestamp'],how='left')

print ("Testing Data+Metadata+Weather Shape {}".format(test.shape))

gc.collect()
le = LabelEncoder()

test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")

test.shape

gc.collect()
test.drop('timestamp',axis=1,inplace=True)
del le

del metadata

del weather_test
test.head()
%%time

# Fill NA

cols = ['year_built', 'air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

for col in cols:

    #train[col].fillna(np.nanmean(train[col].tolist()),inplace=True)

    test[col].fillna(np.nanmean(test[col].tolist()),inplace=True)

    

gc.collect()    


test['square_feet'] = test['square_feet'].astype("uint32")

test['year_built'] = test['year_built'].astype('uint16')

test['air_temperature'] = test['air_temperature'].astype('uint8')

test['cloud_coverage'] = test['cloud_coverage'].astype('uint8')

test['dew_temperature'] = test['dew_temperature'].astype('uint8')

test['precip_depth_1_hr'] = test['precip_depth_1_hr'].astype('uint8')

test['sea_level_pressure'] = test['sea_level_pressure'].astype('uint16')

test['wind_direction'] = test['wind_direction'].astype('uint16')

test['wind_speed'] = test['wind_speed'].astype('uint8')
#test.dtypes
'''%%time

cols = test.columns

imp_mean.fit(test)

imputed_test = pd.DataFrame(data=imp_mean.transform(test), columns = cols)



gc.collect()'''
#del test
test = pd.get_dummies(test, columns=categorical_cols, sparse=True) #imputed_test

test.shape

gc.collect()
test.shape
df = test[train_columns]

gc.collect()
df.shape
%%time

predictions = []

step = 50000

for i in range(0, len(df), step):

    print(i)

    predictions.extend(np.expm1(rf.predict(df.iloc[i: min(i+step, len(df)), :])))
%%time

Submission = pd.DataFrame(test.index,columns=['row_id'])

Submission['meter_reading'] = predictions

Submission['meter_reading'].clip(lower=0,upper=None,inplace=True)

Submission.to_csv("sneaky_weasels.csv",index=None)
Submission.head()