#DATA

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd

import numpy as np

#random forest for regressor

from sklearn.ensemble import RandomForestRegressor

#split features  and target

from sklearn.model_selection import train_test_split

#Pour regression au lieu de accuracy c'est mean_square_error

from sklearn.metrics import mean_squared_error

#Ne pas afficher le warning lors du fit par exemple

#Import for cross_validation

from sklearn.model_selection  import cross_val_score

#import random forest for regression

from sklearn.ensemble import RandomForestRegressor

#Validation

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import KFold

#ignore warnings

import warnings

warnings.filterwarnings('ignore')
# Train File

X_train = pd.read_csv('../input/train.csv', index_col='id')

#Test File

X_test = pd.read_csv('../input/test.csv', index_col='id')

#Submission File

submission = pd.read_csv('../input/sample_submission.csv')
X_train.head()
#a = pd.get_dumnies(X_train['store_and_fwd_flag'], prefix='store_and_fwd_flag', drop_first=True)  

#X_train[dumnies.columns] = dummies

# TRAIN

X_train.loc[X_train['store_and_fwd_flag'] == 'N', 'store_and_fwd_flag'] = 0

X_train.loc[X_train['store_and_fwd_flag'] == 'Y', 'store_and_fwd_flag'] = 1

# TEST

X_test.loc[X_test['store_and_fwd_flag'] == 'N', 'store_and_fwd_flag'] = 0

X_test.loc[X_test['store_and_fwd_flag'] == 'Y', 'store_and_fwd_flag'] = 1
#X_train.head(1)

#X_train = X_train.sort_values(by='trip_duration', ascending=False)
X_train = X_train[(X_train['trip_duration']  < 8000) & (X_train['passenger_count'] >= 1)]





#Gel only trip > 200s ie 3mn

#X_train = X_train[(X_train['trip_duration']  > 200) & (X_train['store_and_fwd_flag'] < 7500)]

#Delete line where trip_duration > 43200 ie 12h and store_and_fwd_flag = 1

#X_train = X_train.drop(X_train[(X_train['trip_duration']  > 43200) & (X_train['store_and_fwd_flag'] == 1)].index)

# Delete Trip < 10mn and store_and_fwd_flag = 0

#X_train = X_train.drop(X_train[(X_train['trip_duration']  <10 ) & (X_train['store_and_fwd_flag'] == 0)].index)

#X_train = X_train[(X_train['trip_duration']  < 43200) & (X_train['store_and_fwd_flag'] != 1 )]

#X_train.sort_values(by='trip_duration', ascending=True)



#X_train['store_and_fwd_flag'].value_counts()
#X_train['passenger_count'].value_counts()
#X_train.describe()
X_test.head(1)
#X_test.describe()
## Fonction features et target

def split_dataset(df, features, target='trip_duration'):

    X = df[features]

    y = df[target]

    return X, y
def date_split(df_train, df_test, date='pickup_datetime'):

    ##X_train

    cols=df_train[date]

    date_cols=pd.to_datetime(cols)

    df_train['year'] = date_cols.dt.year

    df_train['month'] = date_cols.dt.month

    df_train['day'] = date_cols.dt.day

    df_train['hour'] = date_cols.dt.hour

    df_train['minute'] = date_cols.dt.minute

    df_train['second'] = date_cols.dt.second

    #df_train = df_train.drop(['pickup_datetime'], axis=1)

    ##X_test

    cols2=df_test[date]

    date_cols2=pd.to_datetime(cols2)

    df_test['year'] = date_cols2.dt.year

    df_test['month'] = date_cols2.dt.month

    df_test['day'] = date_cols2.dt.day

    df_test['hour'] = date_cols2.dt.hour

    df_test['minute'] = date_cols2.dt.minute

    df_test['second'] = date_cols2.dt.second

    #df_test = df_test.drop(['pickup_datetime'], axis=1)

    return df_train, df_test
#X_train['pickup_datetime']

#date_cols=pd.to_datetime(cols)

#X_train['year'] = date_cols.dt.year

#X_train.head(1)
#Split  date for X_train et X_test

X_train, X_test = date_split(X_train, X_test)
def Get_cols(df, features_test=X_test.columns):

    #get X_test columns in  X_train

    X_train_features = df[features_test]

    return  X_train_features

#Same columns for X_test and X_train

X_trainGet_cols = Get_cols(X_test)

#Get only columns numbers

numbers = X_trainGet_cols.select_dtypes(np.number)

#Definition features and target in the file train

X_train_features, y_train_target = split_dataset(X_train, features=numbers.columns)

#X_test numbers

X_test = X_test.select_dtypes(np.number)



##### OPTIMIZATION #########

#rf = RandomForestRegressor()

#kf = KFold(n_splits=5, random_state=1)                                                                                      

#loses = cross_val_score(rf, X_train_features, y_train_target, cv=kf, scoring='neg_mean_squared_log_error')

# np.sqrt(-loses.mean())

#loses = [np.sqrt(-l) for l in loses]

# np.mean(loses)

#loses[:5]
#Initialise Random Forest regressor

#rf = RandomForestRegressor()

#cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

##Cross Validation

#lossess = -cross_val_score(rf, X_train_features, y_train_target, cv=cv, scoring='neg_mean_squared_log_error')
## FIT all the train set

rf = RandomForestRegressor()

rf.fit(X_train_features, y_train_target)
#X_train_features.head()
#X_test.head()
#X_train_features.head(1)
##### Predict in the training

y_train_pred = rf.predict(X_train_features)



###### Predict in the test

y_test_pred = rf.predict(X_test)



y_test_pred.mean()
#!rm submission.csv
submission["trip_duration"] = y_test_pred

#Convertir notre fichier en csv

submission.to_csv('submission.csv', index=False)