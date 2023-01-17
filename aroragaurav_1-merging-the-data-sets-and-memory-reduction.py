# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/kkbox-churn-prediction-challenge"]).decode("utf8"))

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

import numpy as np # linear alegbra

import pandas as pd # data processing

import os # os commands

from datetime import datetime as dt #work with date time format

import dask.dataframe as dd



%matplotlib inline 

# initiate matplotlib backend

import seaborn as sns # work over matplotlib with improved and more graphs

import matplotlib.pyplot as plt #some easy plotting



transactions = pd.read_csv('../input/kkbox-churn-prediction-challenge/transactions.csv', engine = 'c', sep=',')#reading the transaction file
transactions =transactions.append(pd.read_csv('../input/kkbox-churn-prediction-challenge/transactions.csv', engine = 'c', sep=','))
transactions.info()
transactions.describe()


print("payment_plan_days min: ",transactions['payment_plan_days'].min())

print("payment_plan_days max: ",transactions['payment_plan_days'].max())



print('payment_method_id min:', transactions['payment_method_id'].min())

print('payment_method_id max:', transactions['payment_method_id'].max())

# h=change the type of these series



transactions['payment_method_id'] = transactions['payment_method_id'].astype('int8')

transactions['payment_plan_days'] = transactions['payment_plan_days'].astype('int16')

print('plan list price varies from ', transactions['plan_list_price'].min(), 'to ',transactions['plan_list_price'].max() )

print('actual amount varies from ', transactions['actual_amount_paid'].min(),'to ', transactions['actual_amount_paid'].max() )


transactions['plan_list_price'] = transactions['plan_list_price'].astype('int16')

transactions['actual_amount_paid'] = transactions['actual_amount_paid'].astype('int16')
transactions.info()


transactions['is_auto_renew'] = transactions['is_auto_renew'].astype('int8') # chainging the type to boolean

transactions['is_cancel'] = transactions['is_cancel'].astype('int8')#changing the type to boolean
sum(transactions.memory_usage()/1024**2) # memory usage 


transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'].astype(str), infer_datetime_format = True, exact=False)

# converting the series to string and then to datetime format for easy manipulation of dates

print("memory usage for transaction df is: ", np.round(sum(transactions.memory_usage()/1024**2),2), "GB") # this wouldn't change the size of df as memory occupied by object is similar to datetime
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'].astype(str), infer_datetime_format = True, exact=False)

print("done!")
agg = {'payment_plan_days':['mean','sum', 'count'] , 'payment_method_id':['max','min'],

       'plan_list_price':['mean','sum'], 'actual_amount_paid':['mean','sum'], 'is_auto_renew':['mean','sum'],

       'transaction_date':'min', 'membership_expire_date':'max', 'is_cancel':['mean','sum']}
transactions_train = transactions[transactions['membership_expire_date']<='2017-02-28'];
transactions_test = transactions[(transactions['membership_expire_date']<='2017-03-31')

                                &(transactions['membership_expire_date']>'2017-02-28')]
del transactions
transactions_train = transactions_train.groupby('msno').agg(agg)
transactions_test = transactions_test.groupby('msno').agg(agg)
transactions_test.columns = transactions_test.columns.get_level_values(0)+'_'+transactions_test.columns.get_level_values(1)
transactions_train.columns = transactions_train.columns.get_level_values(0)+'_'+transactions_train.columns.get_level_values(1)
transactions_test.head()
transactions_train.head()
transactions_test = transactions_test.append(transactions_train)
agg = {'payment_plan_days_mean':'mean', 'payment_plan_days_sum':'sum',

       'payment_plan_days_count':'sum', 'payment_method_id_max':'max', 'payment_method_id_min':'min',

       'plan_list_price_mean':'mean', 'plan_list_price_sum':'sum', 

       'actual_amount_paid_mean':'mean', 'actual_amount_paid_sum':'sum', 

       'is_auto_renew_mean':'mean', 'is_auto_renew_sum':'sum',

       'transaction_date_min':'min', 'membership_expire_date_max':'max', 'is_cancel_mean':'mean',

       'is_cancel_sum':'sum'}
transactions_test = transactions_test.groupby(level=0).agg(agg)
transactions_test.head()
print("size of transactions_train is :", sum(transactions_train.memory_usage()/1024**2)) # memory usage 

print("size of transactions_test is :", sum(transactions_test.memory_usage()/1024**2))
'''to make columns to mark if a particular user has chaned its payment method id'''

def payment_method_id_change(df):

    df['payment_method_id_change'] = df['payment_method_id_max'] - df['payment_method_id_min']

    df['payment_method_id_change'] = df['payment_method_id_change'].map(lambda x: 1 if x>0 else 0)

    df.drop(['payment_method_id_max','payment_method_id_min'], inplace=True, axis=1)
payment_method_id_change(transactions_train)

payment_method_id_change(transactions_test)
transactions_test.columns
transactions_train.columns
members = pd.read_csv('../input/kkbox-churn-prediction-challenge/members_v3.csv')
members.info()
members.describe()
members['city']=members['city'].astype('int8');

members['bd'] = members['bd'].astype('int16');

members['bd']=members['bd'].astype('int8');

members['registration_init_time'] = pd.to_datetime(members['registration_init_time'].astype(str), infer_datetime_format = True, exact=False)

#members['expiration_date'] = pd.to_datetime(members['expiration_date'].astype(str), infer_datetime_format = True, exact=False)
print("size of members is :", sum(members.memory_usage()/1024**2))
train = pd.read_csv('../input/kkbox-churn-prediction-challenge/train.csv')

train = train.append(pd.read_csv('../input/kkbox-churn-prediction-challenge/train_v2.csv'))

train.head()
train['is_churn'] = train['is_churn'].astype('int8');
train = train.groupby('msno').max()
train = train.reset_index().merge(transactions_train.reset_index(), how='left', on='msno')
#train = train.merge(transactions, on='msno',how='left', sort= False)
train = train.reset_index().merge(members, how='left',on='msno')
test = pd.read_csv('../input/kkbox-churn-prediction-challenge/sample_submission_v2.csv')
test = test.merge(transactions_test.reset_index(), how='left',on='msno')
test = test.merge(members, how='left',on='msno')
print("Shape of train data is :", train.shape)

print("Shape of test data is :", test.shape)
# deleting the previously imported df as they occupy space in memory

del transactions_test

del transactions_train

del members
#total memory consumptions by all these data frame

print('size of train df is :', np.sum(train.memory_usage()/1024**2))

print('size of test df is :', np.sum(test.memory_usage()/1024**2))
test.to_csv('test_unprocessed')

train.to_csv('train_unprocessed')