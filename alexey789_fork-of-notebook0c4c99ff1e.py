# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



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
members = pd.read_csv('../input/members_v3.csv')
transactions = pd.read_csv('../input/transactions.csv')

transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

#transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())

transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)

transactions = transactions.drop_duplicates(subset=['msno'], keep='first')
#discount

transactions['discount'] = transactions['plan_list_price'] - transactions['actual_amount_paid']

#amt_per_day

transactions['amt_per_day'] = transactions['actual_amount_paid'] / transactions['payment_plan_days']

#is_discount

transactions['is_discount'] = transactions.discount.apply(lambda x: 1 if x > 0 else 0)

#membership_duration

transactions['membership_days'] = pd.to_datetime(transactions['membership_expire_date']).subtract(pd.to_datetime(transactions['transaction_date'])).dt.days.astype(int)

train['is_train'] = 1

test['is_train'] = 0

combined = pd.concat([train, test], axis = 0)



combined = pd.merge(combined, members, how='left', on='msno')

members = []; print('members merge...') 



gender = {'male':1, 'female':2}

combined['gender'] = combined['gender'].map(gender)



combined = pd.merge(combined, transactions, how='left', on='msno')

transactions=[]; print('transaction merge...')



train = combined[combined['is_train'] == 1]

test = combined[combined['is_train'] == 0]



train.drop(['is_train'], axis = 1, inplace = True)

test.drop(['is_train'], axis = 1, inplace = True)



del combined
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

last_user_logs = []



df_iter = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)





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

train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)

test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)



train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)

test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)
train = train.fillna(0)

test = test.fillna(0)



cols = [c for c in train.columns if c not in ['is_churn','msno']]
def xgb_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'log_loss', metrics.log_loss(labels, preds)



fold = 1

for i in range(fold):

    params = {

        'eta': 0.02,

        'min_child_weight': 5,

        'gamma': 0.3,

        'max_depth': 7,

        'objective': 'binary:logistic',

        'eval_metric': 'logloss',

        'seed': 17,

        'silent': True

    }

    x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)

    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

    model = xgb.train(params, xgb.DMatrix(x1, y1), 300,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500

    if i != 0:

        pred1 += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

    else:

        pred1 = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

pred1 /= fold
test['is_churn'] = pred1.clip(0.+1e-15, 1-1e-15)

test[['msno','is_churn']].to_csv('xgbsub2.csv.gz', index=False, compression='gzip')