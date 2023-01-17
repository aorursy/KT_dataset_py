# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

gc.enable()

from multiprocessing import Pool, cpu_count



import xgboost as xgb



from sklearn import *

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

print(train.info())

print(np.max(train['is_churn']))

print(np.min(train['is_churn']))

train['is_churn'] = train['is_churn'].astype(np.int8)

print(train.info())
test=pd.read_csv('../input/sample_submission_zero.csv')

print(np.max(test['is_churn']))

print(np.min(test['is_churn']))


test['is_churn'] = test['is_churn'].astype(np.int8)

print(train.info())
transactions = pd.read_csv('../input/transactions.csv')

print(transactions.info())

print(transactions.head())
transactions['is_cancel'] = transactions['is_cancel'].astype(np.int8)

transactions['payment_method_id'] = transactions['payment_method_id'].astype(np.int8)

transactions['payment_plan_days'] = transactions['payment_plan_days'].astype(np.int16)

transactions['plan_list_price'] = transactions['plan_list_price'].astype(np.int16)

transactions['actual_amount_paid'] = transactions['actual_amount_paid'].astype(np.int16)

transactions['is_auto_renew'] = transactions['is_auto_renew'].astype(np.int8)



print(transactions.info())

print(transactions.head())
transactions['membership_expire_year'] = transactions['membership_expire_date'].apply(lambda x: int(str(x)[:4]))

transactions['membership_expire_month'] = transactions['membership_expire_date'].apply(lambda x: int(str(x)[4:6]))

transactions['membership_expire_day'] = transactions['membership_expire_date'].apply(lambda x: int(str(x)[-2:]))

transactions['transaction_year'] = transactions['transaction_date'] .apply(lambda x: int(str(x)[:4]))

transactions['transaction_month']  = transactions['transaction_date'] .apply(lambda x: int(str(x)[4:6]))

transactions['transaction_day']  = transactions['transaction_date'] .apply(lambda x: int(str(x)[-2:]))

print(transactions.info())
transactions['membership_expire_year'] = transactions['membership_expire_year'].astype(np.int16)

transactions['membership_expire_month'] = transactions['membership_expire_month'].astype(np.int8)

transactions['membership_expire_day'] = transactions['membership_expire_day'].astype(np.int8)

transactions['transaction_year'] = transactions['transaction_year'].astype(np.int16)

transactions['transaction_month'] = transactions['transaction_month'].astype(np.int8)

transactions['transaction_day'] = transactions['transaction_day'].astype(np.int8)

print(transactions.info())
transactions_train = transactions.loc[transactions.transaction_date < 20170201]

transactions_test = transactions.loc[transactions.transaction_date < 20170301]

transactions_train = transactions_train.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)

transactions_test = transactions_test.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)

transactions_train = transactions_train.drop_duplicates(subset=['msno'], keep='first')

transactions_test = transactions_test.drop_duplicates(subset=['msno'], keep='first')

train = pd.merge(train, transactions_train, how='left', on='msno')

test = pd.merge(test, transactions_test, how='left', on='msno')

transactions=[]; transactions_train=[]; transactions_test=[]
print(train.info())

print(train.head())




train = train.drop('transaction_date', 1)

test = test.drop('transaction_date', 1)

train = train.drop('membership_expire_date', 1)

test = test.drop('membership_expire_date', 1)









transactions=[]

print(train.info())

print(test.info())
members = pd.read_csv('../input/members.csv')

members.info()

members.head()
members['city'] = members['city'].astype(np.int8)

members['bd'] = members['bd'].astype(np.int16)

members['registered_via']=members['registered_via'].astype(np.int16)

members['registration_init_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[:4]))

members['registration_init_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))

members['registration_init_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[-2:]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[:4]))

members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))

members['expiration_day'] = members['expiration_date'].apply(lambda x: int(str(x)[-2:]))
members['registration_init_year'] = members['registration_init_time'].astype(np.int16)

members['registration_init_month'] = members['registration_init_time'].astype(np.int8)

members['registration_init_date'] = members['registration_init_time'].astype(np.int8)

members['expiration_year'] = members['expiration_date'].astype(np.int16)

members['expiration_month'] = members['expiration_date'].astype(np.int8)

members['expiration_day'] = members['expiration_date'].astype(np.int8)
members = members.drop('registration_init_time', 1)

members = members.drop('expiration_date', 1)

members = members.drop_duplicates(subset=['msno'], keep='first')

train = pd.merge(train, members, how='left', on='msno')

test = pd.merge(test, members, how='left', on='msno')

train.info()
gender = {'male':1, 'female':2}

train['gender'] = train['gender'].map(gender)

test['gender'] = test['gender'].map(gender)



train = train.fillna(0)

test = test.fillna(0)

print(train.info())

print(test.info())

print(train.head())

train['is_churn'].nunique()

cols = [c for c in train.columns if c not in ['is_churn','msno']]



train.head()



def xgb_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'log_loss', sklearn.metrics.log_loss(labels, preds)



fold = 1

for i in range(fold):

    params = {

        'eta': 0.02, #use 0.002

        'max_depth': 7,

        'objective': 'binary:logistic',

        'eval_metric': 'logloss',

        'seed': i,

        'silent': True

    }

    x1, x2, y1, y2 = sklearn.model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)

    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

    model = xgb.train(params, xgb.DMatrix(x1, y1), 150,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500

    if i != 0:

        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

    else:

        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

pred /= fold

test['is_churn'] = pred.clip(0.0000001, 0.999999)

test[['msno','is_churn']].to_csv('submission3.csv', index=False)


