# This gets 2.08398 on the LB with only XGBoost and no user_log data.

# This is done by adjusting the transaction date distributions between training and test



############################################################

## Imports

############################################################

from datetime import datetime

import numpy as np

import pandas as pd

from xgboost import XGBClassifier



############################################################

## other globals

############################################################

PATH = "../input/"

gender = {'male':1, 'female':2}



############################################################

## function definitions

############################################################

## This gets X/Y train/test data

def get_xy(train_data, test_data, features, y_feature='is_churn', istest=False, dropOutliers=False, outlierMinMax=(-0.4, 0.418), replaceNan=True, fold=1, rs=1):

    if istest:

        if replaceNan:

            X = train_data[features].fillna(-999).values

            Y = train_data[y_feature].values

        else:

            X = train_data[features].values

            Y = train_data[y_feature].values

        if dropOutliers:

            X = X[ Y  > outlierMinMax[0] ]

            Y = Y[ Y  > outlierMinMax[0] ]

            X = X[ Y  < outlierMinMax[1] ]

            Y = Y[ Y  < outlierMinMax[1] ]

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=10, random_state=rs, shuffle=True)

        folds = list(kf.split(train_data))

        train_index,test_index = tuple(list(folds)[fold])

        x_train_m, x_test_m = X[train_index], X[test_index]

        y_train_m, y_test_m = Y[train_index], Y[test_index]

    else:

        train_index, test_index = 0, 0

        if replaceNan:

            x_train_m = train_data[features].fillna(-999).values

            x_test_m = test_data[features].fillna(-999).values

        else:

            x_train_m = train_data[features].values

            x_test_m = test_data[features].values

        y_train_m = train_data[y_feature].values

        y_test_m = np.nan

        if dropOutliers:

            x_train_m = x_train_m[ y_train_m  > outlierMinMax[0] ]

            y_train_m = y_train_m[ y_train_m  > outlierMinMax[0] ]

            x_train_m = x_train_m[ y_train_m  < outlierMinMax[1] ]

            y_train_m = y_train_m[ y_train_m  < outlierMinMax[1] ]

    return x_train_m, x_test_m, y_train_m, y_test_m, train_index, test_index

 

def get_last_df(df, date_field='date'):

    df = df.sort_values(by=[date_field], ascending=[False]).reset_index(drop=True)

    df = df.drop_duplicates(subset=['msno'], keep='first')

    return df



############################################################

## loading data

############################################################

print('Loading data...')

sample = pd.read_csv(PATH + 'sample_submission_zero.csv')

train = pd.read_csv(PATH + 'train.csv')

members = pd.read_csv(PATH + 'members.csv')

trans = pd.read_csv(PATH + 'transactions.csv')

test = sample.copy()



############################################################

## Feature engineering

############################################################

print('Feature engineering...')



print(' Members...')

## Add duration

members['registration_duration_month'] = (pd.to_datetime(members.expiration_date, format='%Y%m%d') - pd.to_datetime(members.registration_init_time, format='%Y%m%d'))/ np.timedelta64(1, 'M')



## Add Gender

members['gender'] = members['gender'].map(gender)



print(' Transactions...')

trans = get_last_df(trans,'transaction_date')



############################################################

## Join Data

############################################################

print('Join Data...')

train = pd.merge(train, members, how='left', on='msno')

test = pd.merge(test, members, how='left', on='msno')

 

## ** try adding 1 month to train, to make it similar to test?

## This was determined by looking at the percentile values of the train vs test

train['expiration_date'] = train['expiration_date'] + 100

 

train = pd.merge(train, trans, how='left', on='msno')

test = pd.merge(test, trans, how='left', on='msno')

 

feature_names = [feature for feature in train.columns[2:]]

test = test[feature_names]



############################################################

## create X, Y

############################################################

print('create X, Y...')

istest=False

x_train_xgb, x_test_xgb, y_train_xgb, y_test_xgb, train_index, test_index = get_xy(train, test, feature_names, istest=istest, replaceNan=False, dropOutliers=False)



############################################################

## Model Parameters

############################################################

print('Setting up model parameters')

params_xgb = {}

params_xgb['random_state'] = 1

params_xgb['learning_rate'] = 0.037

params_xgb['max_depth'] = 7

params_xgb['objective'] = 'binary:logistic'

params_xgb['n_estimators']=242

params_xgb['gamma']=0

params_xgb['subsample']=1

params_xgb['reg_alpha']=0

params_xgb['reg_lambda']=1

params_xgb['base_score']=np.mean(train.is_churn)

 

############################################################

## Run Models

############################################################

print('Running Models')

 

model_xgb = XGBClassifier(**params_xgb).fit(x_train_xgb, y_train_xgb)

y_hat_xgb = model_xgb.predict(x_test_xgb)



############################################################

## Results

############################################################

print('Calculating Results...')

results = pd.DataFrame()

sample = pd.read_csv(PATH + 'sample_submission_zero.csv')

 

results = sample.copy()

results['is_churn']=y_hat_xgb



results.to_csv('submission.{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index = False, float_format = '%.0f')

print('Done!')