# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#Import necessary packages for data analysis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing, linear_model, metrics

import gc; gc.enable()

import seaborn as sns

sns.set(style = 'whitegrid', color_codes = True)

%matplotlib inline

#Machine Learning Algorithms

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}

data = {

    'train_1': pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date']),

    'test': pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date']),

    'items': pd.read_csv('../input/items.csv'),

    'stores': pd.read_csv('../input/stores.csv'),

    'trans': pd.read_csv('../input/transactions.csv', parse_dates=['date']),

    'holidays': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),

    'oil': pd.read_csv('../input/oil.csv', parse_dates=['date']),

    }
train = data['train_1'][(data['train_1']['date'].dt.month == 8) & (data['train_1']['date'].dt.day > 15)]

del data['train_1']; gc.collect();

target = train['unit_sales'].values

target[target < 0.] = 0.

train['unit_sales'] = np.log1p(target)



def df_lbl_enc(df):

    for c in df.columns:

        if df[c].dtype == 'object':

            lbl = preprocessing.LabelEncoder()

            df[c] = lbl.fit_transform(df[c])

            print(c)

    return df



def df_transform(df):

    df['date'] = pd.to_datetime(df['date'])

    df['yea'] = df['date'].dt.year

    df['mon'] = df['date'].dt.month

    df['day'] = df['date'].dt.day

    df['date'] = df['date'].dt.dayofweek

    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})

    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})

    df = df.fillna(-1)

    return df

#Items data

data['items'] = df_lbl_enc(data['items'])

train = pd.merge(train, data['items'], how='left', on=['item_nbr'])

test = pd.merge(data['test'], data['items'], how='left', on=['item_nbr'])

del data['test']; gc.collect();

del data['items']; gc.collect();

#Transactions data

train = pd.merge(train, data['trans'], how='left', on=['date','store_nbr'])

test = pd.merge(test, data['trans'], how='left', on=['date','store_nbr'])

del data['trans']; gc.collect();

target = train['transactions'].values

target[target < 0.] = 0.00015

train['transactions'] = np.log1p(target)

#Stores data

data['stores'] = df_lbl_enc(data['stores'])

train = pd.merge(train, data['stores'], how='left', on=['store_nbr'])

test = pd.merge(test, data['stores'], how='left', on=['store_nbr'])

del data['stores']; gc.collect();

#Holidays data

data['holidays'] = data['holidays'][data['holidays']['locale'] == 'National'][['date','transferred']]

data['holidays']['transferred'] = data['holidays']['transferred'].map({'False': 0, 'True': 1})

train = pd.merge(train, data['holidays'], how='left', on=['date'])

test = pd.merge(test, data['holidays'], how='left', on=['date'])

del data['holidays']; gc.collect();

#Oil data

train = pd.merge(train, data['oil'], how='left', on=['date'])

test = pd.merge(test, data['oil'], how='left', on=['date'])

del data['oil']; gc.collect();

#Join data

train = df_transform(train)

test = df_transform(test)

col = [c for c in train if c not in ['id', 'unit_sales','perishable','transactions']]

x1 = train[(train['yea'] != 2016)]

x2 = train[(train['yea'] == 2016)]

del train; gc.collect();

y1 = x1['transactions'].values

y2 = x2['transactions'].values
def NWRMSLE(y, pred, w):

    return metrics.mean_squared_error(y, pred, sample_weight=w)**0.5



col = [c for c in x1 if c not in ['id', 'unit_sales','city','cluster','perishable']]

y1 = x1['unit_sales'].values

y2 = x2['unit_sales'].values



r3 = RandomForestRegressor(n_estimators=79, max_depth=5, n_jobs=-1, 

                                    verbose=0, warm_start=True)



r4 = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate = 0.05, 

                                       verbose=0, warm_start=True,

                                       subsample= 0.65, max_features = 0.35)

#Fit a random forest classifier to our training set

r3.fit(x1[col], y1)

r4.fit(x1[col], y1)

a3 = NWRMSLE(y1, r3.predict(x1[col]), x1['perishable'])

a4 = NWRMSLE(y1, r4.predict(x1[col]), x1['perishable'])

print('model fit')
#Output file

N3 = str(a3)

print('Accuracy = ',a3*100)

test['unit_sales'] = r3.predict(test[col])

cut = 0.+1e-12 # 0.+1e-15

test['unit_sales'] = (np.exp(test['unit_sales']) - 1).clip(lower=cut)



output_file = 'sample_submission_RF1.csv'

 

test[['id','unit_sales']].to_csv(output_file, index=False, float_format='%.2f')

print('file created')