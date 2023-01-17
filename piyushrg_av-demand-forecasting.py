# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random

random_seed = 0

random.seed(random_seed)

np.random.seed(random_seed)
def RMSLE(actual, predicted):



    predicted = np.array([np.log(np.abs(x+1.0)) for x in predicted])  # doing np.abs for handling neg values  

    actual = np.array([np.log(np.abs(x+1.0)) for x in actual])

    log_err = actual-predicted

    

    return 1000*np.sqrt(np.mean(log_err**2))
train = pd.read_csv('/kaggle/input/demand-forecasting/train_0irEZ2H.csv')

test = pd.read_csv('/kaggle/input/demand-forecasting/test_nfaJ3J5.csv')

submit = pd.read_csv('/kaggle/input/demand-forecasting/sample_submission_pzljTaX.csv')
train.head(2)
print(train.shape, test.shape, submit.shape)
train['store_sku'] = (train['store_id'].astype('str') + "_" + train['sku_id'].astype('str'))

test['store_sku'] = (test['store_id'].astype('str') + "_" + test['sku_id'].astype('str'))

len(train['store_sku'].unique()) - len(test['store_sku'].unique()) # checking if the combination of store and sku is same across train and test.
assert len(np.intersect1d(train['store_sku'].unique(), test['store_sku'].unique())) == len(test['store_sku'].unique())
train.info()
temp = train[train['total_price'].isnull()]['base_price']

train['total_price'] = train['total_price'].fillna(temp)
#Appending train and test together for faster manipulation of data

test['units_sold'] = -1

data = train.append(test, ignore_index = True)
data.info()
print('Checking Data distribution for Train! \n')

for col in train.columns:

    print(f'Distinct entries in {col}: {train[col].nunique()}')

    print(f'Common # of {col} entries in test and train: {len(np.intersect1d(train[col].unique(), test[col].unique()))}')
data.describe()
train.units_sold.describe()
(train[train.units_sold <= 200].units_sold).hist()
train['units_sold'].hist()
np.log1p(train['units_sold']).hist()
data[['base_price', 'total_price']].plot.box()
# Making price based new features



train['diff'] = train['base_price'] - train['total_price']



train['relative_diff_base'] = train['diff']/train['base_price']

train['relative_diff_total'] = train['diff']/train['total_price']



train.head(2)
test['diff'] = test['base_price'] - test['total_price']

test['relative_diff_base'] = test['diff']/test['base_price']

test['relative_diff_total'] = test['diff']/test['total_price']

test.head(2)
# Studying correlation between features and the target. This will help us in regression later.

cols = ['base_price', 'total_price', 'diff', 'relative_diff_base', 'relative_diff_total'

        , 'is_featured_sku', 'is_display_sku', 'units_sold']

train[cols].corr().loc['units_sold']
print(f'current # of features in cols: {len(cols)}')

cols.remove('units_sold')

print(f'current # of features to be used: {len(cols)}')
from sklearn.model_selection import train_test_split



X = train[cols]

y = np.log1p(train['units_sold']) # Transforming target into normal via logarithmic operation



Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape)
Xtrain.isnull().sum()
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(Xtrain, ytrain)
preds = reg.predict(Xval)

print(f'The validation RMSLE error for baseline model is: {RMSLE(np.exp(yval), np.exp(preds))}')
sub_preds = reg.predict(test[cols])

submit['units_sold'] = np.exp(sub_preds)

submit.head(2)
submit.to_csv('sub_baseline_v1.csv', index = False)
from category_encoders import TargetEncoder, MEstimateEncoder

encoder = MEstimateEncoder()

encoder.fit(train['store_id'], train['units_sold'])

train['store_encoded'] = encoder.transform(train['store_id'], train['units_sold'])

test['store_encoded'] = encoder.transform(test['store_id'], test['units_sold'])
encoder.fit(train['sku_id'], train['units_sold'])

train['sku_encoded'] = encoder.transform(train['sku_id'], train['units_sold'])

test['sku_encoded'] = encoder.transform(test['sku_id'], test['units_sold'])
skus = train.sku_id.unique()

print(skus[:2])



test_preds = test.copy()

test_preds.tail(2)
def sku_model(sku, cols_to_use, reg):

    X = train[train['sku_id'] == sku][cols_to_use]

    y = train[train['sku_id'] == sku]['units_sold']

    

    Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size = 0.2, random_state = 1)

    reg.fit(X,np.log1p(y))

    

    y_pred = reg.predict(Xval)

    err = RMSLE(yval, np.exp(y_pred))

    print(f'RMSLE for {sku} is: {err}')

    

    preds = reg.predict(test[test['sku_id'] == sku][cols_to_use])    

    temp_df =  pd.DataFrame.from_dict({'record_ID': test_preds[test_preds['sku_id'] == sku]['record_ID'],

                                       'units_sold':  np.exp(preds)})

    return err, temp_df
cols_to_use = cols + ['store_encoded', 'sku_encoded']
err = dict() # for documenting error for each sku type

sub = pd.DataFrame(None, columns = ['record_ID', 'units_sold'])

reg = RandomForestRegressor(random_state = 2288)



for sku in skus:

    err[sku], temp = sku_model(sku, cols_to_use, reg)

    sub = sub.append(temp)



print(np.mean(list(err.values())))
sub.sort_values(by = ['record_ID']).to_csv('sub_sku_RF_v2.csv', index = False)
import lightgbm as lgb

from lightgbm import LGBMRegressor

from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold, StratifiedKFold
cols_to_use
cols_to_use += ['store_id', 'sku_id']

# For defining categorical features to the model, we will build `cat_cols`

cat_cols = ['is_featured_sku', 'is_display_sku', 'store_id', 'sku_id']
X = train[cols_to_use]

y = np.log1p(train['units_sold']) # Transforming target into normal via logarithmic operation



Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape)
Xtest = test[cols_to_use]
def runLGB(Xtrain, ytrain, Xval, yval, cat_cols, Xtest = None):

    params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'l1',

    #'max_depth': 9, 

    'learning_rate': 0.1

    ,'verbose': 1

    , "min_data_in_leaf" : 10

    }



    n_estimators = 800

    early_stopping_rounds = 10



    d_train = lgb.Dataset(Xtrain.copy(), label=ytrain.copy(), categorical_feature=cat_cols)

    d_valid = lgb.Dataset(Xval.copy(), label=yval.copy(), categorical_feature=cat_cols)

    watchlist = [d_train, d_valid]



    model = lgb.train(params, d_train, n_estimators

                      , valid_sets = [d_train, d_valid]

                      , verbose_eval=n_estimators

                      , early_stopping_rounds=early_stopping_rounds)



    preds = model.predict(Xval, num_iteration=model.best_iteration)

    err = RMSLE(yval, np.exp(preds))

    

    preds_test = model.predict(Xtest, num_iteration=model.best_iteration)

    return  preds, err, np.exp(preds_test), model
pred_val, err, pred_test,model = runLGB(Xtrain, ytrain, Xval, yval, cat_cols, Xtest)
submit['units_sold'] = pred_test

submit.to_csv('lgb_sub_store_sku_v3.csv', index = False)
a =model.feature_importance(importance_type='split')

feature = pd.DataFrame(model.feature_name(), columns = ['feature'])

feature['importance'] = a

feature = feature.sort_values(by = ['importance'], ascending = False)

feature.head(11)
from datetime import datetime

train['week'] = train['week'].astype('str')

train['week'] = [datetime.strptime(x, '%d/%m/%y') for x in train['week']]
train.head()
test['week'] = test['week'].astype('str')

test['week'] = [datetime.strptime(x, '%d/%m/%y') for x in test['week']]
import datetime

train['weekend_date'] = [x + datetime.timedelta(days=6) for x in train['week']]

test['weekend_date'] = [x + datetime.timedelta(days=6) for x in test['week']]
current_cols = list(train.columns)
import datetime 

def extract_time_features(df):

    

    start_date = datetime.datetime(2011,1, 17)

    

    print('starting basic feature extraction for week start date!')



    df['year'] = df['week'].dt.year

    df['date'] = [x.day for x in df['week']]

    df['month'] = df['week'].dt.month

    df['weekday'] = df['week'].dt.dayofweek

    df['weeknum'] = df['week'].dt.weekofyear

    

    df['week_serial']  = [divmod((x-start_date).total_seconds(), 86400)[0]/7 for x in df['week']]

    



    '''

    print('starting month end related feature extraction for week start date!')



    df['quarter'] = [x.quarter for x in df['week']]

    df['is_month_start'] = [x.is_month_start for x in df['week']]

    df['is_month_end'] = [x.is_month_end for x in df['week']]

    df['is_month_start'] = df['is_month_start'].astype(int)

    df['is_month_end'] = df['is_month_end'].astype(int)

    

    df['start_week']= df.assign(start_week=pd.cut(df.date,[0,9,15,23,31],labels=[1,2,3,4]))['start_week']

    df['start_week'] = df['start_week'].astype(int)

    '''



    print('Starting basic feature extraction for week end date!')

    

    df['end_year'] = df['weekend_date'].dt.year

    df['end_date'] = [x.day for x in df['weekend_date']]

    df['end_month'] = df['weekend_date'].dt.month

    df['end_weekday'] = df['weekend_date'].dt.dayofweek

    df['end_weeknum'] = df['weekend_date'].dt.weekofyear

    df['end_week_serial']  = [divmod((x-start_date).total_seconds(), 86400)[0]/7 for x in df['weekend_date']]



    '''

    print('starting month end related feature extraction for week start date!')



    df['end_quarter'] = [x.quarter for x in df['weekend_date']]

    df['end_is_month_start'] = [x.is_month_start for x in df['weekend_date']]

    df['end_is_month_end'] = [x.is_month_end for x in df['weekend_date']]

    df['end_is_month_start'] = df['end_is_month_start'].astype(int)

    df['end_is_month_end'] = df['end_is_month_end'].astype(int)

    

    df['end_week'] = df.assign(end_week=pd.cut(df.end_date,[0,9,15,23,31],labels=[1,2,3,4]))['end_week']

    df['end_week'] = df['end_week'].astype(int)

    '''

    return df
train = extract_time_features(train)

train.tail()
train.info()
test = extract_time_features(test)

test.tail()
def Diff(li1, li2): 

    return list(set(li1) - set(li2))



total_cols = list(test.columns)

new_feat = Diff(total_cols, current_cols)

new_feat
train[new_feat].info()
print(f'The number of features used before: {len(cols_to_use)}')

print(f'The number of categorical features used before: {len(cat_cols)}')
cols_to_use += new_feat

#cat_cols += new_feat



print(f'The number of features to be used now: {len(cols_to_use)}')

print(f'The number of categorical features to be used now: {len(cat_cols)}')
cols_to_use
Xtest = test[cols_to_use]
X = train[cols_to_use]

y = np.log1p(train['units_sold']) # Transforming target into normal via logarithmic operation



Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape)
pred_val, err, pred_test,model = runLGB(Xtrain, ytrain, Xval, yval, cat_cols, Xtest)
submit['units_sold'] = pred_test

submit.to_csv('lgb_time_store_sku_v4.csv', index = False)
a =model.feature_importance(importance_type='split')

feature = pd.DataFrame(model.feature_name(), columns = ['feature'])

feature['importance'] = a

feature = feature.sort_values(by = ['importance'], ascending = False)

feature.head(20)
# making changes in the LGB model to improve the predictions (Tuning!)

def runLGB2(Xtrain, ytrain, Xval, yval, cat_cols, Xtest = None):

    params = {

    'boosting_type': 'dart', #dropout aided regressive trees (DART) # improves accuracy

    'objective': 'regression',

    'metric': 'l1', 

    #'max_depth': 10, 

    'learning_rate': 0.5

    ,'verbose': 1

    }

    

    #regularising for overfitting with inf depth

    params["min_data_in_leaf"] = 15 

    params["bagging_fraction"] = 0.7

    params["feature_fraction"] = 0.7

    #params["bagging_freq"] = 3

    params["bagging_seed"] = 50



    n_estimators = 575

    early_stopping_rounds = 30



    d_train = lgb.Dataset(Xtrain.copy(), label=np.log1p(ytrain.copy()), categorical_feature=cat_cols)

    d_valid = lgb.Dataset(Xval.copy(), label=np.log1p(yval.copy()), categorical_feature=cat_cols)

    watchlist = [d_train, d_valid]



    model = lgb.train(params, d_train, n_estimators

                      , valid_sets = [d_train, d_valid]

                      , verbose_eval=125

                      , early_stopping_rounds=early_stopping_rounds)



    preds = model.predict(Xval, num_iteration=model.best_iteration)

    err = RMSLE(yval['units_sold'], np.exp(preds))

    

    preds_test = np.exp(model.predict(Xtest, num_iteration=model.best_iteration))

    return  preds, err, preds_test, model
encoder.fit(train[new_feat], train['units_sold'])

train[new_feat] = encoder.transform(train[new_feat], train['units_sold'])

test[new_feat] = encoder.transform(test[new_feat], test['units_sold'])
import time



preds_buff = 0

err_buff = []



X = train[cols_to_use]

y = train[['units_sold']]



n_splits = 10

kf = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=22)



for dev_index, val_index in kf.split(X, y):

    start = time.time()

    Xtrain, Xval = X.iloc[dev_index], X.iloc[val_index]

    ytrain, yval = y.iloc[dev_index], y.iloc[val_index]    

    

    pred_val, err, pred_test,model = runLGB2(Xtrain, ytrain, Xval, yval, cat_cols, Xtest)

    preds_buff += pred_test

    err_buff.append(err)

    print(f'Mean Error: {np.mean(err_buff)}; Split error: {err}')

    print(f'Total time in seconds for this fold: {time.time()-start}')

    print('\n')



preds_buff /= n_splits
print(err_buff, np.mean(err_buff))
submit['units_sold'] = np.abs(preds_buff)

submit.to_csv('lgb_time_10cv_v5.csv', index = False)
a =model.feature_importance(importance_type='split')

feature = pd.DataFrame(model.feature_name(), columns = ['feature'])

feature['importance'] = a

feature = feature.sort_values(by = ['importance'], ascending = False)

feature.head(20)
submit.head()