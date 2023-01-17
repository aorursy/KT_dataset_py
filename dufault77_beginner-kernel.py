import pandas as pd

import numpy as np

import matplotlib as plt

import seaborn as sns

import math

import keras





%matplotlib inline

dftrain = pd.read_csv("../Documents/3rd Year Sem 1/energy/train.csv")

dfsample = pd.read_csv("../Documents/3rd Year Sem 1/energy/sample_submission.csv")

dfbuilding = pd.read_csv("../Documents/3rd Year Sem 1/energy/building_metadata.csv")

dftest = pd.read_csv("../Documents/3rd Year Sem 1/energy/test.csv")

dfweathertrain = pd.read_csv("../Documents/3rd Year Sem 1/energy/weather_train.csv")

dfweathertest = pd.read_csv("../Documents/3rd Year Sem 1/energy/weather_test.csv")
dftrain.info()
dftrain.head(5)
dfsample.head(5)
dfbuilding.info()
dfbuilding.head(5)
dftest.head(5)
dfweathertrain.info()
dfweathertrain.head(5)
dfweathertest.head(5)
sns.heatmap(dftrain.isnull(), yticklabels = False,cmap="Greens")
dftrain["meter"].isnull().sum() + dftrain["meter_reading"].isnull().sum()
sns.heatmap(dfbuilding.isnull(), yticklabels = False,cmap="Greens")
sns.heatmap(dfweathertrain.isnull(), yticklabels = False,cmap="Greens")
sns.heatmap(dftrain.corr(), cmap='coolwarm',annot=True)
sns.distplot(dfweathertrain["air_temperature"].dropna(), color='green').set_title("air_temperature", fontsize=16)

sns.distplot(dfweathertrain["cloud_coverage"].dropna(), color='green').set_title("cloud_coverage", fontsize=16)

sns.distplot(dfweathertrain["dew_temperature"].dropna(), color='green').set_title("dew_temperature", fontsize=16)
sns.distplot(dfweathertrain["precip_depth_1_hr"].dropna(), color='green').set_title("precip_depth_1_hr", fontsize=16)
sns.distplot(dfbuilding["year_built"].dropna(), color='blue').set_title("year_built", fontsize=16)

sns.distplot(dfbuilding["square_feet"].dropna(), color='blue').set_title("square_feet", fontsize=16)

#These non-numerical cateogories must have their types changed

dftrain['timestamp'] = pd.to_datetime(dftrain['timestamp'])

dftest['timestamp'] = pd.to_datetime(dftest['timestamp'])

dfweathertrain['timestamp'] = pd.to_datetime(dfweathertrain['timestamp'])

dfweathertest['timestamp'] = pd.to_datetime(dfweathertest['timestamp'])

dfbuilding['primary_use'] = dfbuilding['primary_use'].astype('category')    

#Creating age feature

dfbuilding['age'] = dfbuilding['year_built'].max() - dfbuilding['year_built']
#Dealing with missing values by filling with negative values





dfweathertrain['cloud_coverage'] = dfweathertrain['cloud_coverage'].fillna(-1)

dfweathertest['cloud_coverage'] = dfweathertest['cloud_coverage'].fillna(-1)



dfweathertrain['precip_depth_1_hr'] = dfweathertrain['precip_depth_1_hr'].fillna(-1)

dfweathertest['precip_depth_1_hr'] = dfweathertest['precip_depth_1_hr'].fillna(-1)



dfweathertrain['precip_depth_1_hr'] = dfweathertrain['precip_depth_1_hr'].fillna(-1)

dfweathertest['precip_depth_1_hr'] = dfweathertest['precip_depth_1_hr'].fillna(-1)



dfbuilding['floor_count'] = dfbuilding['floor_count'].fillna(-1)



dfbuilding['age'] = dfbuilding['age'].fillna(-1)



#Dropping unneccesary columns

dfbuilding.drop(['year_built'],inplace = True,axis =1)



dfweathertrain.drop(['sea_level_pressure'],inplace = True,axis=1)

dfweathertest.drop(['sea_level_pressure'],inplace = True,axis=1)

dfweathertrain.drop(['wind_direction'],inplace = True,axis=1)

dfweathertest.drop(['wind_direction'],inplace = True,axis=1)
temp_df = dftrain[['building_id']]

temp_df = temp_df.merge(dfbuilding, on=['building_id'], how='left')

del temp_df['building_id']

dftrain = pd.concat([dftrain, temp_df], axis=1)



temp_df = dftest[['building_id']]

temp_df = temp_df.merge(dfbuilding, on=['building_id'], how='left')



del temp_df['building_id']

dftest = pd.concat([dftest, temp_df], axis=1)

del temp_df
temp_df = dftrain[['site_id','timestamp']]

temp_df = temp_df.merge(dfweathertrain, on=['site_id','timestamp'], how='left')



del temp_df['site_id'], temp_df['timestamp']

dftrain = pd.concat([dftrain, temp_df], axis=1)



temp_df = dftest[['site_id','timestamp']]

temp_df = temp_df.merge(dfweathertest, on=['site_id','timestamp'], how='left')



del temp_df['site_id'], temp_df['timestamp']

dftest = pd.concat([dftest, temp_df], axis=1)



del temp_df
dftrain.head(5)
del dfbuilding

del dfweathertest

del dfweathertrain
from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()

dftrain['primary_use'] = dftrain['primary_use'].astype(str)

dftrain['primary_use'] = labels.fit_transform(dftrain['primary_use']).astype(np.int8)



dftest['primary_use'] = dftest['primary_use'].astype(str)

dftest['primary_use'] = labels.fit_transform(dftest['primary_use']).astype(np.int8)
 
#Creating time based features

dftrain['hourofday'] = dftrain['timestamp'].dt.hour.astype(np.int8)  

dftrain['dayofweek'] = dftrain['timestamp'].dt.dayofweek.astype(np.int8)

dftrain['dayofmonth'] = dftrain['timestamp'].dt.day.astype(np.int8)

dftrain['monthofyear'] = dftrain['timestamp'].dt.month.astype(np.int8)

dftrain['weekofyear'] = dftrain['timestamp'].dt.weekofyear.astype(np.int8)

dftrain['dayofyear'] = dftrain['timestamp'].dt.dayofyear.astype(np.int16)

    



dftest['hourofday'] = dftest['timestamp'].dt.hour.astype(np.int8)

dftest['dayofweek'] = dftest['timestamp'].dt.dayofweek.astype(np.int8)

dftest['dayofmonth'] = dftest['timestamp'].dt.day.astype(np.int8) 

dftest['monthofyear'] = dftest['timestamp'].dt.month.astype(np.int8)

dftest['weekofyear'] = dftest['timestamp'].dt.weekofyear.astype(np.int8)

dftest['dayofyear'] = dftest['timestamp'].dt.dayofyear.astype(np.int16)

    

dftrain = dftrain.drop(["timestamp"], axis=1)

dftest = dftest.drop(["timestamp"], axis=1)
dftrain.head(5)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dftrain.drop('meter_reading',axis=1), 

                                                    dftrain['meter_reading'], test_size=0.30, 

                                                    random_state=101)
from sklearn.preprocessing import MinMaxScaler

# Data needs to be scaled to a small range like 0 to 1 for the neural

# network to work well.

scaler = MinMaxScaler(feature_range=(0, 1))



# Scale both the training inputs and outputs

scaled_training = scaler.fit_transform(dftrain)

scaled_testing = scaler.transform(dftest)
scaled_training_df = pd.DataFrame(scaled_training, columns=dftrain.columns.values)

scaled_testing_df = pd.DataFrame(scaled_testing, columns=dftest.columns.values)
dftrain.head(5)
scaled_training_df.head(5)
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(dftrain.drop('meter_reading',axis=1), 

#                                                   dftrain['meter_reading'], test_size=0.30, 

#                                                    random_state=101)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_training_df.drop('meter_reading',axis=1), 

                                                    scaled_training_df['meter_reading'], test_size=0.30, 

                                                    random_state=101)
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(50, input_dim=18, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss="mean_squared_error", optimizer="adam")
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(X_train, y_train,validation_split=0.2, epochs=1)
#Exploring Data

model.evaluate(X_test, y_test, verbose=0)