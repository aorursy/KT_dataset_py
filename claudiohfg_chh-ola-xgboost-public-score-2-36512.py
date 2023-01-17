# Fix FutureWarning Messages in scikit-learn

# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)



import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from datetime import datetime, timedelta

from dateutil.parser import parse as dt_parse

from collections import Counter



from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import mean_squared_error



import xgboost as xgb



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!head /kaggle/input/chh-ola/train.csv
!sed 's/\+AF8-//g' /kaggle/input/chh-ola/train.csv > train.csv

!sed 's/_//g' /kaggle/input/chh-ola/test.csv > test.csv
class Ut:

  @staticmethod

  def to_timestamp(dt):

    return dt_parse(dt, dayfirst=False).timestamp()



  @staticmethod

  def flag_to_num(vl):

    if vl == 'N':

      return 0

    else:

      return 1

  

  @staticmethod

  def to_float(vl):

    try:

      if type(vl) == type('str'):

        idx = vl.find('-')

        if idx != -1:

          txt = vl.split('-')

          return float(txt[1])

      return float(vl)

    except:

      print(vl)

      return float(0)



  @staticmethod

  def rmse(predictions, targets):

    """Data convertion and RMSE calculation"""

    return np.sqrt(mean_squared_error(np.exp(predictions), np.exp(targets)))
# Load training data file

train_set = pd.read_csv('train.csv', low_memory=False, dtype=str)

train_set.dropna(inplace=True)

train_set.reset_index(drop=True, inplace=True)



# Load testing data file

test_set = pd.read_csv('test.csv', low_memory=False, dtype=str)

test_set['totalamount'] = 0



# Create a column to distinguish one from the other

train_set['PROPOSITO'] = 1

test_set['PROPOSITO'] = 0



# Concatenate all data into one dataframe

all_set = pd.concat([train_set, test_set], ignore_index=True)



# Delete train_set and test_set to free memory

del(train_set)

del(test_set)



# MTA Tax should be the same in all rides

all_set['mtatax'] = 0.5



# Data conversion

# Y or N to 1 or 0

all_set['storedflag'] = all_set['storedflag'].apply(Ut.flag_to_num)

# Conversion to datetime format

all_set['pickuptime'] = all_set['pickuptime'].apply(Ut.to_timestamp)

all_set['droptime'] = all_set['droptime'].apply(Ut.to_timestamp)

# Conversion to float

all_set['drivertip'] = all_set['drivertip'].apply(Ut.to_float)

all_set['mtatax'] = all_set['mtatax'].apply(Ut.to_float)

all_set['tollamount'] = all_set['tollamount'].apply(Ut.to_float)

all_set['extracharges'] = all_set['extracharges'].apply(Ut.to_float)

all_set['improvementcharge'] = all_set['improvementcharge'].apply(Ut.to_float)

all_set['totalamount'] = all_set['totalamount'].apply(Ut.to_float)
# Total time

all_set['totaltime'] = all_set['droptime'] - all_set['pickuptime']

# All Taxes

all_set['taxes'] = all_set['drivertip'] + all_set['mtatax'] + all_set['tollamount'] + all_set['extracharges'] + all_set['improvementcharge']
features_cat = ['vendorid', 'paymentmethod', 'ratecode', 'storedflag']

features_num = ['drivertip', 'pickuploc', 'droploc', 'mtatax', 'distance', 'pickuptime', 'droptime', 'numpassengers', 

                'tollamount', 'extracharges', 'improvementcharge', 'totalamount', 'totaltime', 'taxes']

target = 'totalamount'



# Numerical features will be converted to float.

for col in features_num:

    all_set[col] = all_set[col].astype(float)



# Categorical features will be converted to float then to string.    

for col in features_cat:

    all_set[col] = all_set[col].astype(float)

    all_set[col] = all_set[col].astype(str)



all_set['PROPOSITO'] = all_set['PROPOSITO'].astype(int)

all_set['ID'] = all_set['ID'].astype(int)
# Convert categorical variable into dummy/indicator variables

all_dum = pd.get_dummies(all_set)



train_df = all_dum[all_dum['PROPOSITO'] == 1].copy()

test_df = all_dum[all_dum['PROPOSITO'] == 0].copy()



del(all_dum)



train_df.drop(columns=['PROPOSITO'], inplace=True)

test_df.drop(columns=['PROPOSITO'], inplace=True)



# Remove rows where pickup loc is equal to drop loc

train_df = train_df[train_df['pickuploc'] != train_df['droploc']]

# Drop rows where total amount is 0, since there are no free rides

train_df = train_df[train_df['totalamount'] > 0]

test_df.drop(columns=[target], inplace=True)
# Alongside hyperparameter searching, I also did a feature searching to check 

# which combination would provide better results

features_to_keep = [

  'taxes',

  'pickuploc',

  'ratecode_2.0',

  'ratecode_1.0',

  'ratecode_5.0',

  'storedflag_0.0',

  'ratecode_4.0',

  'totaltime',

  'ratecode_3.0',

  'droploc',

  'numpassengers',

  'distance',

  'storedflag_1.0',

  'vendorid_2.0',

  'paymentmethod_1.0',

  'vendorid_1.0',

  'paymentmethod_2.0'

  ]



X = train_df[features_to_keep].copy()

y = train_df[target].copy()



normalizer = Normalizer()

norm_X = normalizer.fit_transform(X)

y = np.log(y)
if False:

  params = {

      'colsample_bytree':[0.9], 

      'gamma':[0.3],

      'max_depth': [9],  

      'min_child_weight':[2], 

      'subsample':[0.9],

      'n_estimators': [50],

      'objective': ['reg:squarederror'],

      'n_jobs': [8],

      }



  # Initialize XGB and GridSearch

  eval_model = xgb.XGBRegressor(nthread=-1) 



  grid = GridSearchCV(eval_model, params, cv=2)

  grid.fit(train_X, train_y)



  pred_y = grid.predict(test_X)

  print('RMSE Test = ', Ut.rmse(pred_y, test_y))



  print(grid)

  print(grid.best_params_)
if False:

  params = {

      'min_child_weight': st.randint(2, 9), 

      'gamma': st.uniform(0.1, 0.9),  

      'subsample': st.uniform(0.1, 0.9),

      'colsample_bytree': st.uniform(0.1, 0.9), 

      'max_depth': st.randint(3, 9),

      'n_estimators': [50],

      'objective': ['reg:squarederror'],

      # 'n_jobs': [8],

      }



  eval_model = xgb.XGBRegressor(nthread=-1) 



  grid = RandomizedSearchCV(eval_model, params, cv=2, n_jobs=1, n_iter=10)

  grid.fit(train_X, train_y)



  pred_y = grid.predict(test_X)

  print('RMSE Test = ', Ut.rmse(pred_y, test_y))



  print(grid)
# Besides hyperparameter searching, it is also important to see how the results change 

# when we remove features that have a poor correlation to the "total amount".

if False:

    train_correlation = train_df.corr()

    train_correlation = train_correlation['totalamount'].apply(abs)

    train_correlation = train_correlation.sort_values(na_position='first')

    train_correlation = pd.DataFrame(train_correlation).reset_index()

    train_correlation.dropna(inplace=True)

    

    best_rmse = math.inf

    best_idx = 0



    features_to_analyse = train_correlation['index'].unique()

    for idx in range(len(features_to_analyse) - 1):

        features = features_to_analyse[idx:]

        dX = train_df[features].copy()

        dy = train_df[target].copy()



        normalizer = Normalizer()

        norm_dX = normalizer.fit_transform(dX)

        dy = np.log(dy)



        train_dX, test_dX, train_dy, test_dy = train_test_split(norm_dX, dy, test_size=0.2, random_state=42)



        params = {

            'objective': 'reg:squarederror',

            'n_estimators': 100,

            'subsample': 0.9, 

            'min_child_weight': 1, 

            'max_depth': 9,

            'gamma': 0.3, 

            'colsample_bytree': 0.9,

            'n_jobs': 8,

            'verbose_eval':'False',

        }

        model = xgb.XGBRegressor(**params)

        model.fit(train_dX,train_dy)



        pred_dy = model.predict(test_dX)

        print('Features cut = ', idx, ' Resulting RMSE = ', Ut.rmse(pred_dy, test_dy))

        

        if best_rmse > rmse_res:

            best_rmse = rmse_res

            best_idx = idx



    print('Features cut = ', best_idx, ' Resulting RMSE = ', best_rmse)

    features_to_keep = features_to_analyse[best_idx:]
train_X, test_X, train_y, test_y = train_test_split(norm_X, y, test_size=0.2, random_state=42)



# These hyperparameters are the result of the searchings of Step 9

params = {

    'objective': 'reg:squarederror',

    'n_estimators': 1000,

    'subsample': 0.9, 

    'min_child_weight': 1, 

    'max_depth': 9,

    'gamma': 0.3, 

    'colsample_bytree': 0.9,

    'n_jobs': 8,

    'verbose_eval':'False',

}

model = xgb.XGBRegressor(**params)

model.fit(train_X,train_y)



pred_y = model.predict(test_X)

print('RMSE Test = ', Ut.rmse(pred_y, test_y))
real_X = normalizer.transform(test_df[features_to_keep].copy())



model = xgb.XGBRegressor(**params)

model.fit(norm_X, y)



predictions = np.exp(model.predict(real_X))
# Have a glimpse of the predictions

pd.DataFrame(predictions).head(20)
result = []

for idx in range(test_df.shape[0]):

  result.append([idx, predictions[idx]])

result = pd.DataFrame(result, columns=['ID', 'total_amount'])

result.to_csv('result.csv', index=False)
# Delete temp files so they're not taken as results by mistake

!rm train.csv

!rm test.csv