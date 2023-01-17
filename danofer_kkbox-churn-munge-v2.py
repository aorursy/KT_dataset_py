!ls ../input
from multiprocessing import Pool, cpu_count

import gc; gc.enable()

import xgboost as xgb

import pandas as pd

import numpy as np

from sklearn import *

import sklearn
train = pd.read_csv('../input/train.csv')

train = pd.concat((train, pd.read_csv('../input/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

test = pd.read_csv('../input/sample_submission_v2.csv')



transactions = pd.read_csv('../input/transactions.csv', usecols=['msno'])

transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)

transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())

transactions.columns = ['msno','trans_count']

train = pd.merge(train, transactions, how='left', on='msno')

test = pd.merge(test, transactions, how='left', on='msno')

transactions = []; print('transaction merge...')



user_logs = pd.read_csv('../input/user_logs_v2.csv', usecols=['msno'])

#user_logs = pd.read_csv('../input/user_logs.csv', usecols=['msno'])

#user_logs = pd.concat((user_logs, pd.read_csv('../input/user_logs_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)

user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())

user_logs.columns = ['msno','logs_count']

train = pd.merge(train, user_logs, how='left', on='msno')

test = pd.merge(test, user_logs, how='left', on='msno')

user_logs = []; print('user logs merge...')



members = pd.read_csv('../input/members_v3.csv')

train = pd.merge(train, members, how='left', on='msno')

test = pd.merge(test, members, how='left', on='msno')

members = []; print('members merge...') 
gender = {'male':1, 'female':2}

train['gender'] = train['gender'].map(gender)

test['gender'] = test['gender'].map(gender)



train = train.fillna(0)

test = test.fillna(0)
transactions = pd.read_csv('../input/transactions.csv')

transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)

transactions = transactions.drop_duplicates(subset=['msno'], keep='first')



train = pd.merge(train, transactions, how='left', on='msno')

test = pd.merge(test, transactions, how='left', on='msno')

transactions=[]
def transform_df(df):

    df = pd.DataFrame(df)

    df = df.sort_values(by=['date'], ascending=[False])

    df = df.reset_index(drop=True)

    df = df.drop_duplicates(subset=['msno'], keep='first')

    return df



def transform_df2(df):

    df = df.sort_values(by=['date'], ascending=[False])

    df = df.reset_index(drop=True)

    df = df.drop_duplicates(subset=['msno'], keep='first')

    return df



df_iter = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)

last_user_logs = []

i = 0 #~400 Million Records - starting at the end but remove locally if needed

for df in df_iter:

    if i>35:

        if len(df)>0:

            print(df.shape)

            p = Pool(cpu_count())

            df = p.map(transform_df, np.array_split(df, cpu_count()))   

            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)

            df = transform_df2(df)

            p.close(); p.join()

            last_user_logs.append(df)

            print('...', df.shape)

            df = []

    i+=1

last_user_logs.append(transform_df(pd.read_csv('../input/user_logs_v2.csv')))

last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)

last_user_logs = transform_df2(last_user_logs)



train = pd.merge(train, last_user_logs, how='left', on='msno')

test = pd.merge(test, last_user_logs, how='left', on='msno')

last_user_logs=[]
print(train.shape)

train.head()
pd.to_datetime(train.tail(9).transaction_date,format="%Y%m%d",errors="coerce")
# train.registration_init_time = pd.to_datetime(train.registration_init_time,format="%Y%m%d",errors="coerce")

# test.registration_init_time = pd.to_datetime(test.registration_init_time,format="%Y%m%d",errors="coerce")



# train.registration_init_time.tail()
date_cols = ["transaction_date","membership_expire_date","date","registration_init_time"]
test[date_cols].tail()
# pd.to_datetime(train.transaction_date,format="%Y%m%d")

# pd.to_datetime(test.transaction_date,format="%Y%m%d")



# pd.to_datetime(train.membership_expire_date,format="%Y%m%d")

# pd.to_datetime(test.membership_expire_date,format="%Y%m%d")



# pd.to_datetime(test.date,format="%Y%m%d.0")
# pd.to_datetime(train.tail().registration_init_time.astype(int),format="%Y%m%d",errors="coerce")

# pd.to_datetime(train.tail().date.fillna(0).astype(int),format="%Y%m%d",errors="coerce")
### try to coerce? 



train.transaction_date = pd.to_datetime(train.transaction_date,format="%Y%m%d")

test.transaction_date = pd.to_datetime(test.transaction_date,format="%Y%m%d")



train.membership_expire_date = pd.to_datetime(train.membership_expire_date,format="%Y%m%d")

test.membership_expire_date = pd.to_datetime(test.membership_expire_date,format="%Y%m%d")



train.registration_init_time = pd.to_datetime(train.registration_init_time.astype(int),format="%Y%m%d",errors="coerce")

test.registration_init_time = pd.to_datetime(test.registration_init_time.astype(int),format="%Y%m%d",errors="coerce")



train.date = pd.to_datetime(train.date.fillna(0).astype(int),format="%Y%m%d",errors="coerce")

test.date = pd.to_datetime(test.date.fillna(0).astype(int),format="%Y%m%d",errors="coerce")
train.head()
# train["sum_nan"] = train.isnull().sum(axis=1)

# train["sum_nan"].describe()
train["played_songs_nonUnique_ratio"] = (train['num_100'] + train['num_25'] + train['num_50'] + train['num_75'] + train['num_985'])/train["num_unq"]

test["played_songs_nonUnique_ratio"] = (test['num_100'] + test['num_25'] + test['num_50'] + test['num_75'] + test['num_985'])/test["num_unq"]



train["played_songs_nonUnique_ratio"].describe()
train["price_paid_diff"]  = train.plan_list_price -  train.actual_amount_paid

test["price_paid_diff"]  = test.plan_list_price -  test.actual_amount_paid
# train = train.fillna(0)

# test = test.fillna(0)



# cols = [c for c in train.columns if c not in ['is_churn','msno']]
# def xgb_score(preds, dtrain):

#     labels = dtrain.get_label()

#     return 'log_loss', metrics.log_loss(labels, preds)



# fold = 1

# for i in range(fold):

#     params = {

#         'eta': 0.02, #use 0.002

#         'max_depth': 7,

#         'objective': 'binary:logistic',

#         'eval_metric': 'logloss',

#         'seed': i,

#         'silent': True

#     }

#     x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)

#     watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

#     model = xgb.train(params, xgb.DMatrix(x1, y1), 150,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500

#     if i != 0:

#         pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

#     else:

#         pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

# pred /= fold

# test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)

# test[['msno','is_churn']].to_csv('submission.csv', index=False)

# #test[['msno','is_churn']].to_csv('submission.csv.gz', index=False, compression='gzip')
# import matplotlib.pyplot as plt

# import seaborn as sns

# %matplotlib inline



# plt.rcParams['figure.figsize'] = (7.0, 7.0)

# xgb.plot_importance(booster=model); plt.show()
train.to_csv("kkbox_churn_train_v3.csv.gz",index=False,compression="gzip")
test.to_csv("kkbox_churn_test_v3.csv.gz",index=False,compression="gzip")
# transactions = pd.read_csv('../input/transactions.csv',nrows=1000)

# transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

# transactions_msno_counts = pd.DataFrame(transactions['msno'].value_counts().reset_index()).rename(columns={"msno":"msno_counts"})

# print(transactions_msno_counts.shape)



# transactions.membership_expire_date = pd.to_datetime(transactions.membership_expire_date,format="%Y%m%d")

# transactions.transaction_date = pd.to_datetime(transactions.transaction_date,format="%Y%m%d")



# print(transactions.shape)

# transactions.head()
# transactions["price_paid_diff"]  = transactions.plan_list_price -  transactions.actual_amount_paid
# transactions["transactions_expiry_transaction_days_diff"] = ((transactions.membership_expire_date - transactions.transaction_date).dt.days) 
# transactions["transactions_expiry_transaction_days_diff_div_plan"] = transactions.payment_plan_days/transactions.transactions_expiry_transaction_days_diff
# transactions["transactions_expiry_transaction_days_diff"].describe()

# transactions["transactions_expiry_transaction_days_diff_div_plan"].describe()
# (transactions["transactions_expiry_transaction_days_diff"] != transactions.payment_plan_days).sum()
# transactions.to_csv("kkbox_churn_transactions_v3.csv.gz",index=False,compression="gzip")
def get_date_diffs(df):

    """

    Get time between the expiry date and other columns in days + add day of week, month features.

     - Could add more, e.g. time between other dates, is weekend, etc' .

     membership_expire_date is the deciding date for churn determination.

     # https://www.kaggle.com/danofer/kkbox-churn-getting-started

    """

    df["exp-registration-diff"] = (df.membership_expire_date - df.registration_init_time ).dt.days

    df["exp-transaction-diff"] = (df.membership_expire_date - df.transaction_date ).dt.days

    df["exp-expiration-diff"] = (df.membership_expire_date - df.expiration_date ).dt.days

    df["exp-logdate-diff"] = (df.membership_expire_date - df["date"] ).dt.days

    

    for col in dateCols:

        df["dayOfWeek_%s" %(col)] = df[col].dt.dayofweek

        df["dayOfMonth_%s" %(col)] = df[col].dt.day

    

    df["payment_plan_days_div-exp-expiration-diff"] = df.payment_plan_days / df["exp-expiration-diff"]

    df["payment_plan_days_div-exp-transaction-diff"] = df.payment_plan_days / df["exp-transaction-diff"]

    df["payment_plan_days_div-eexp-logdate-diff"] = df.payment_plan_days / df["exp-logdate-diff"]