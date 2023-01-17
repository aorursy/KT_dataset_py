# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Reduce the memory usage - Inspired by Panchajanya Banerjee

def reduce_mem_usage(df, verbose=True):

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
train = reduce_mem_usage(pd.read_csv("../input/train.csv", parse_dates=["first_active_month"]))

merch = reduce_mem_usage(pd.read_csv("../input/merchants.csv"))

test = reduce_mem_usage(pd.read_csv("../input/test.csv", parse_dates= ["first_active_month"]))

histTrans = reduce_mem_usage(pd.read_csv("../input/historical_transactions.csv"))

newMerchTrans = reduce_mem_usage(pd.read_csv("../input/new_merchant_transactions.csv"))
train.shape
train.head()
import gc

gc.collect()
merch.head()
test.head()
histTrans.head()
newMerchTrans.head()
train.target.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.figure(figsize=(12,5))

plt.hist(train.target.values, bins = 300)

plt.title('Histogram of target Variable')

plt.xlabel('target')

plt.ylabel('counts')

plt.show()
histTrans.head()
histTrans['merchant_category_id'].value_counts(dropna = False)
histTrans['subsector_id'].value_counts(dropna = False)
histTrans['state_id'].value_counts(dropna = False)
histTrans['city_id'].value_counts(dropna = False)
newMerchTrans.head()
newMerchTrans.installments.value_counts()
train.shape
#train_new = train.merge(histTrans, on= 'card_id' ,how = 'left')
newMerchTrans.head()
histTrans.head()
merch.head()
plt.figure(figsize=(12,5))

plt.hist(train.feature_1.values,100)

plt.title('Histogram of feature_1 Variable')

plt.xlabel('feature_1')

plt.ylabel('counts')

plt.show()
train.feature_1.value_counts()
plt.figure(figsize=(12,6))

plt.hist(train.feature_2.values,bins = 100)

plt.title("Histogram of feature_2 variable")

plt.xlabel("feature_2")

plt.ylabel('counts')

plt.show()
train.feature_2.value_counts()
plt.figure(figsize=(12,6))

plt.hist(train.feature_3, bins = 100)

plt.title('histogram of feature_3')

plt.xlabel('feature_3')

plt.ylabel('counts')

plt.show()
train.feature_3.value_counts()
histTrans.head()
train.first_active_month.value_counts().sort_index().tail()
train.first_active_month.value_counts().sort_index().head()
import datetime

#train['first_active_month'] = pd.to_datetime(train['first_active_month'], format = '%Y-%m' )



train['month'] = train["first_active_month"].dt.month

train['year'] = train["first_active_month"].dt.year



test['month'] = test["first_active_month"].dt.month

test['year'] = test["first_active_month"].dt.year



train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days # the last date entry is 2018-02

test['elapsed_time'] = (datetime.date(2018, 1, 1) - test['first_active_month'].dt.date).dt.days # the last date entry is 2018-01
train.info()
test.info()
train.shape
#train = pd.get_dummies(train, columns = ['feature_1','feature_2'])

#test = pd.get_dummies(test, columns = ['feature_1','feature_2'])
train.columns
train.info()
histTrans.head()
null_df = pd.DataFrame(columns = ['columns','Count'])

null_df['Count'] = [histTrans[x].isnull().sum() for x in histTrans.columns ]

null_df['columns'] =histTrans.columns

null_df
histTrans['category_1'].value_counts(dropna = False)
histTrans.columns
histTrans['category_1'] = pd.factorize(histTrans['category_1'])[0]

#histTrans = pd.get_dummies(histTrans, columns = ['category_3', 'category_2'])

histTrans['authorized_flag'] = pd.factorize(histTrans['authorized_flag'])[0]

histTrans['category_3'] = histTrans['category_3'].map({'A':0, 'B':1, 'C':2})
histTrans['category_2'].fillna(histTrans['category_2'].mode()[0], inplace = True)

histTrans['category_3'].fillna(histTrans['category_3'].mode()[0], inplace = True)
#histTrans = pd.get_dummies(histTrans, columns = ['category_3'])
histTrans.head()
sum(histTrans['purchase_amount']<0)
histTrans.shape
#histTrans['month_lag'] = abs(histTrans['month_lag'])
#histTrans['city_id'] = abs(histTrans['city_id'])
train.head()
import gc
gc.collect()
#Feature Engineering - Adding new features inspired by Chau's first kernel

histTrans['purchase_date'] = pd.to_datetime(histTrans['purchase_date'])

histTrans['year'] = histTrans['purchase_date'].dt.year

histTrans['weekofyear'] = histTrans['purchase_date'].dt.weekofyear

histTrans['month'] = histTrans['purchase_date'].dt.month

histTrans['dayofweek'] = histTrans['purchase_date'].dt.dayofweek

histTrans['weekend'] = (histTrans.purchase_date.dt.weekday >=5).astype(int)

histTrans['hour'] = histTrans['purchase_date'].dt.hour 

histTrans['quarter'] = histTrans['purchase_date'].dt.quarter

histTrans['is_month_start'] = histTrans['purchase_date'].dt.is_month_start

histTrans['month_diff'] = ((datetime.datetime.today() - histTrans['purchase_date']).dt.days)//30

histTrans['month_diff'] += histTrans['month_lag']
histTrans['is_month_start'] = pd.factorize(histTrans['is_month_start'])[0]
histTrans.head(10)
#histTrans.head(8)
#histTrans['category_2_mean'].value_counts(dropna = False)
# additional features

histTrans['price'] = histTrans['purchase_amount'] / histTrans['installments']



#Christmas : December 25 2017

histTrans['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-histTrans['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

#Mothers Day: May 14 2017

histTrans['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-histTrans['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

#fathers day: August 13 2017

histTrans['fathers_day_2017']=(pd.to_datetime('2017-08-13')-histTrans['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

#Childrens day: October 12 2017

histTrans['Children_day_2017']=(pd.to_datetime('2017-10-12')-histTrans['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

#Valentine's Day : 12th June, 2017

histTrans['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-histTrans['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

#Black Friday : 24th November 2017

histTrans['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - histTrans['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)



#2018

#Mothers Day: May 13 2018

histTrans['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-histTrans['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)



gc.collect()
histTrans = reduce_mem_usage(histTrans)
gc.collect()
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
# Taking Reference from Other Kernels

def aggregate_transaction_hist(trans, prefix):  

        

    agg_func = {

        'purchase_date' : ['max','min'],

        'month_diff' : ['mean', 'min', 'max', 'var'],

        'weekend' : ['sum', 'mean'],

        'authorized_flag': ['sum', 'mean'],

        'category_1': ['sum','mean', 'max','min'],

        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],

        'installments': ['sum', 'mean', 'max', 'min', 'std'],  

        'month_lag': ['max','min','mean','var'],

        'card_id' : ['size'],

        'month': ['nunique'],

        'hour': ['nunique'],

        'weekofyear': ['nunique'],

        'dayofweek': ['nunique'],

        'year': ['nunique'],

        'subsector_id': ['nunique'],

        'merchant_category_id' : ['nunique'],

        'Christmas_Day_2017' : ['mean'],

        'Mothers_Day_2017' : ['mean'],

        'fathers_day_2017' : ['mean'],

        'Children_day_2017' : ['mean'],

        'Valentine_Day_2017' : ['mean'],

        'Black_Friday_2017' : ['mean'],

        'Mothers_Day_2018' : ['mean'],

        'category_2': ['sum','mean', 'max','min'],

        'category_3': ['sum','mean', 'max','min']

    }

    

    agg_trans = trans.groupby(['card_id']).agg(agg_func)

    agg_trans.columns = [prefix + '_'.join(col).strip() 

                           for col in agg_trans.columns.values]

    agg_trans.reset_index(inplace=True)

    

    df = (trans.groupby('card_id')

          .size()

          .reset_index(name='{}transactions_count'.format(prefix)))

    

    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')

    

    return agg_trans
merge_trans = aggregate_transaction_hist(histTrans, prefix='hist_')

merge_trans.head()
del histTrans

gc.collect()
#merge_trans = aggregate_transaction_hist(histTrans, prefix='hist_')

train = pd.merge(train, merge_trans, on='card_id',how='left')

test = pd.merge(test, merge_trans, on='card_id',how='left')

del merge_trans

gc.collect()
train.head(10)
train.shape
test.shape
#Feature Engineering - Adding new features inspired by Chau's first kernel

train['hist_purchase_date_max'] = pd.to_datetime(train['hist_purchase_date_max'])

train['hist_purchase_date_min'] = pd.to_datetime(train['hist_purchase_date_min'])

train['hist_purchase_date_diff'] = (train['hist_purchase_date_max'] - train['hist_purchase_date_min']).dt.days

train['hist_purchase_date_average'] = train['hist_purchase_date_diff']/train['hist_card_id_size']

train['hist_purchase_date_uptonow'] = (datetime.datetime.today() - train['hist_purchase_date_max']).dt.days

train['hist_purchase_date_uptomin'] = (datetime.datetime.today() - train['hist_purchase_date_min']).dt.days

train['hist_first_buy'] = (train['hist_purchase_date_min'] - train['first_active_month']).dt.days



for feature in ['hist_purchase_date_max','hist_purchase_date_min']:

    train[feature] = train[feature].astype(np.int64) * 1e-9

gc.collect()
#Feature Engineering - Adding new features inspired by Chau's first kernel

test['hist_purchase_date_max'] = pd.to_datetime(test['hist_purchase_date_max'])

test['hist_purchase_date_min'] = pd.to_datetime(test['hist_purchase_date_min'])

test['hist_purchase_date_diff'] = (test['hist_purchase_date_max'] - test['hist_purchase_date_min']).dt.days

test['hist_purchase_date_average'] = test['hist_purchase_date_diff']/test['hist_card_id_size']

test['hist_purchase_date_uptonow'] = (datetime.datetime.today() - test['hist_purchase_date_max']).dt.days

test['hist_purchase_date_uptomin'] = (datetime.datetime.today() - test['hist_purchase_date_min']).dt.days



test['hist_first_buy'] = (test['hist_purchase_date_min'] - test['first_active_month']).dt.days



for feature in ['hist_purchase_date_max','hist_purchase_date_min']:

    test[feature] = test[feature].astype(np.int64) * 1e-9



gc.collect()
# Taking Reference from Other Kernels

def aggregate_transaction_new(trans, prefix):  

        

    agg_func = {

        'purchase_date' : ['max','min'],

        'month_diff' : ['mean', 'min', 'max', 'var'],

        'weekend' : ['sum', 'mean'],

        'authorized_flag': ['sum', 'mean'],

        'category_1': ['sum','mean', 'max','min'],

        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],

        'installments': ['sum', 'mean', 'max', 'min', 'std'],  

        'month_lag': ['max','min','mean','var'],

        'card_id' : ['size'],

        'month': ['nunique'],

        'hour': ['nunique'],

        'weekofyear': ['nunique'],

        'dayofweek': ['nunique'],

        'year': ['nunique'],

        'subsector_id': ['nunique'],

        'merchant_category_id' : ['nunique']

    }

    

    agg_trans = trans.groupby(['card_id']).agg(agg_func)

    agg_trans.columns = [prefix + '_'.join(col).strip() 

                           for col in agg_trans.columns.values]

    agg_trans.reset_index(inplace=True)

    

    df = (trans.groupby('card_id')

          .size()

          .reset_index(name='{}transactions_count'.format(prefix)))

    

    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')

    

    return agg_trans
newMerchTrans['authorized_flag'] = newMerchTrans['authorized_flag'].map({'Y': 1, 'N': 0})

newMerchTrans['category_1'] = newMerchTrans['category_1'].map({'Y': 1, 'N': 0})
#Feature Engineering - Adding new features inspired by Chau's first kernel

newMerchTrans['purchase_date'] = pd.to_datetime(newMerchTrans['purchase_date'])

newMerchTrans['year'] = newMerchTrans['purchase_date'].dt.year

newMerchTrans['weekofyear'] = newMerchTrans['purchase_date'].dt.weekofyear

newMerchTrans['month'] = newMerchTrans['purchase_date'].dt.month

newMerchTrans['dayofweek'] = newMerchTrans['purchase_date'].dt.dayofweek

newMerchTrans['weekend'] = (newMerchTrans.purchase_date.dt.weekday >=5).astype(int)

newMerchTrans['hour'] = newMerchTrans['purchase_date'].dt.hour 

newMerchTrans['quarter'] = newMerchTrans['purchase_date'].dt.quarter

newMerchTrans['is_month_start'] = newMerchTrans['purchase_date'].dt.is_month_start

newMerchTrans['month_diff'] = ((datetime.datetime.today() - newMerchTrans['purchase_date']).dt.days)//30

newMerchTrans['month_diff'] += newMerchTrans['month_lag']



gc.collect()



#impute missing values

#newMerchTrans['category_2'] = newMerchTrans['category_2'].fillna(1.0,inplace=True)

#newMerchTrans['category_3'] = newMerchTrans['category_3'].fillna('A',inplace=True)

#newMerchTrans['merchant_id'] = newMerchTrans['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)



newMerchTrans['category_3'] = newMerchTrans['category_3'].map({'A':0, 'B':1, 'C':2}) 



gc.collect()
merge_new = aggregate_transaction_new(newMerchTrans, prefix='new_')

del newMerchTrans

gc.collect()


train = pd.merge(train, merge_new, on='card_id',how='left')

test = pd.merge(test, merge_new, on='card_id',how='left')

del merge_new



gc.collect()
#Feature Engineering - Adding new features inspired by Chau's first kernel

train['new_purchase_date_max'] = pd.to_datetime(train['new_purchase_date_max'])

train['new_purchase_date_min'] = pd.to_datetime(train['new_purchase_date_min'])

train['new_purchase_date_diff'] = (train['new_purchase_date_max'] - train['new_purchase_date_min']).dt.days

train['new_purchase_date_average'] = train['new_purchase_date_diff']/train['new_card_id_size']

train['new_purchase_date_uptonow'] = (datetime.datetime.today() - train['new_purchase_date_max']).dt.days

train['new_purchase_date_uptomin'] = (datetime.datetime.today() - train['new_purchase_date_min']).dt.days

train['new_first_buy'] = (train['new_purchase_date_min'] - train['first_active_month']).dt.days

for feature in ['new_purchase_date_max','new_purchase_date_min']:

    train[feature] = train[feature].astype(np.int64) * 1e-9



#Feature Engineering - Adding new features inspired by Chau's first kernel

test['new_purchase_date_max'] = pd.to_datetime(test['new_purchase_date_max'])

test['new_purchase_date_min'] = pd.to_datetime(test['new_purchase_date_min'])

test['new_purchase_date_diff'] = (test['new_purchase_date_max'] - test['new_purchase_date_min']).dt.days

test['new_purchase_date_average'] = test['new_purchase_date_diff']/test['new_card_id_size']

test['new_purchase_date_uptonow'] = (datetime.datetime.today() - test['new_purchase_date_max']).dt.days

test['new_purchase_date_uptomin'] = (datetime.datetime.today() - test['new_purchase_date_min']).dt.days

test['new_first_buy'] = (test['new_purchase_date_min'] - test['first_active_month']).dt.days

for feature in ['new_purchase_date_max','new_purchase_date_min']:

    test[feature] = test[feature].astype(np.int64) * 1e-9

    

#added new feature - Interactive

train['card_id_total'] = train['new_card_id_size'] + train['hist_card_id_size']

train['purchase_amount_total'] = train['new_purchase_amount_sum'] + train['hist_purchase_amount_sum']



test['card_id_total'] = test['new_card_id_size'] + test['hist_card_id_size']

test['purchase_amount_total'] = test['new_purchase_amount_sum'] + test['hist_purchase_amount_sum']



gc.collect()
#Check for Missing Values after Concatination



obs = train.isnull().sum().sort_values(ascending = False)

percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)

pd.concat([obs, percent], axis = 1,keys= ['Number of Observations', 'Percent'])
del train['first_active_month']

del test['first_active_month']
train_id = train['card_id']

test_id = test['card_id']
del train['card_id']

del test['card_id']
train = train.apply(lambda x: x.fillna(x.median()))

test = test.apply(lambda x: x.fillna(x.median()))
train['card_id'] = train_id

test['card_id'] = test_id 
# Remove the Outliers if any 

train['outliers'] = 0

train.loc[train['target'] < -30, 'outliers'] = 1

train['outliers'].value_counts()
train.shape, test.shape
train.to_csv("train_processed2.csv",index = False)

test.to_csv("test_processed2.csv",index = False)
#train = train.drop(['card_id', 'first_active_month'], axis = 1)

#test = test.drop(['card_id', 'first_active_month'], axis = 1)
#train.category_1
# Get the X and Y

'''df_train_columns = [c for c in train.columns if c not in ['target','outliers']] 

target = train['target']

del train['target']'''
#len(df_train_columns)
test.shape
from sklearn.ensemble import RandomForestRegressor
#model = RandomForestRegressor(n_estimators=1000, criterion = 'mse', random_state=1, n_jobs = -1)
#model.fit(train,target)
#prediction = model.predict(test)
'''sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission['target'] = prediction

sample_submission.to_csv('simple_RF.csv', index=False)'''