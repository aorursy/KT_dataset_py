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
train = pd.read_csv('/kaggle/input/demand-forecasting/train_0irEZ2H.csv')

test = pd.read_csv('/kaggle/input/demand-forecasting/test_nfaJ3J5.csv')

submit = pd.read_csv('/kaggle/input/demand-forecasting/sample_submission_pzljTaX.csv')
train.head()
print(train.shape, test.shape, submit.shape)
train['store_sku'] = (train['store_id'].astype('str') + "_" + train['sku_id'].astype('str'))

test['store_sku'] = (test['store_id'].astype('str') + '_' + train['sku_id'].astype('str'))
train['store_sku'].head()
test['store_sku'].head()
len(train['store_sku'].unique()) - len(test['store_sku'].unique()) ## checking if the combination of store and sku is same across train and test.
len(np.intersect1d(train['store_sku'].unique(), test['store_sku'].unique())) == len(test['store_sku'].unique())
assert len(np.intersect1d(train['store_sku'].unique(), test['store_sku'].unique())) == len(test['store_sku'].unique())
train.info()
temp = train[train['total_price'].isnull()]['base_price']

temp
train['total_price'] = train['total_price'].fillna(temp)
#Appending train and test together for faster manipulation of data

test['units_sold'] = -1

data = train.append(test, ignore_index = True)
test.head()
data.head()
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
#Making price based new features



train['diff']=train['base_price']-train['total_price']

train['relative_diff_base'] = train['diff']/train['base_price']

train['relative_diff_total'] = train['diff']/train['total_price']



train.head()
test['diff']=test['base_price']-test['total_price']

test['relative_diff_base'] = test['diff']/test['base_price']

test['relative_diff_total'] = test['diff']/test['total_price']



test.head()
#Studying correlation between features and target variable



train.columns

cols = ['total_price', 'base_price',

       'is_featured_sku', 'is_display_sku', 'units_sold', 'store_sku', 'diff',

       'relative_diff_base', 'relative_diff_total']

train[cols].corr()
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
def RMSLE(actual, predicted):



    predicted = np.array([np.log(np.abs(x+1.0)) for x in predicted])  # doing np.abs for handling neg values  

    actual = np.array([np.log(np.abs(x+1.0)) for x in actual])

    log_err = actual-predicted

    

    return 1000*np.sqrt(np.mean(log_err**2))
preds = reg.predict(Xval)

print(f'The validation RMSLE error for baseline model is : {RMSLE(np.exp(yval), np.exp(preds))}')
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
skus = train.sku_id.unique() #unique sku_ids

print(skus)
skus = train.sku_id.unique() #unique sku_ids

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
cols_to_use
err = dict() # for documenting error for each sku type

sub = pd.DataFrame(None, columns = ['record_ID', 'units_sold'])

reg = RandomForestRegressor(random_state = 2288)



for sku in skus:

    err[sku], temp = sku_model(sku, cols_to_use, reg)

    sub = sub.append(temp)



print(np.mean(list(err.values())))
sub.sort_values(by = ['record_ID']).to_csv('sub_sku_RF_v2.csv', index = False)