# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_v2.csv')

transactions = pd.read_csv('../input/transactions_v2.csv')

members = pd.read_csv('../input/members_v3.csv')
#user_logs = pd.read_csv('../input/user_logs.csv')

transactions.head(1)
print(transactions.msno.nunique())

print(train.msno.nunique())
train_transactions = pd.merge(train, transactions, on='msno', how='left')

train_transactions.head(1)
composite = pd.merge(train_transactions, members, on='msno', how='left')

composite.head(1)
print(composite.msno.nunique())

composite.shape
composite['membership_expire_date'] = pd.to_datetime(composite['membership_expire_date'], format='%Y%m%d')

composite['transaction_date'] = pd.to_datetime(composite['transaction_date'], format='%Y%m%d')
mask = composite['registration_init_time'].notnull()

composite.loc[composite[mask].index, 'registration_init_time'

             ] = composite.loc[composite[mask].index, 'registration_init_time'].astype('int')
composite.head(1)
composite.isnull().sum()
composite.shape
for col in transactions.columns:

    print(col + ' has ' + str(transactions[col].nunique()) + ' unique_values.')
#transactions = transactions.drop_duplicates()

#transactions['payment_method_id', 'payment_plan_days']
#def churn_rate(df, col):

#    col_rate = df.groupby(col)[['is_churn']].mean().reset_index().sort_values(

#        'is_churn', ascending=False)

#    sns.barplot(x=col, y='is_churn', data=col_rate)

#    print(plt.show())

#    return col_rate
sns.kdeplot(transactions.payment_plan_days)
transactions['payment_plan_days'].hist(bins=20)
sns.distplot(transactions.payment_plan_days)
sns.distplot(transactions.plan_list_price)
sns.distplot(transactions.actual_amount_paid)
composite.payment_plan_days.value_counts().head(10)
composite.plan_list_price.value_counts().head(10)
composite.actual_amount_paid.value_counts().head(10)
composite['discount'] = (composite.plan_list_price - composite.actual_amount_paid)
composite.discount.value_counts()
my_list = ['plan_list_price', 'actual_amount_paid', 'payment_plan_days','is_churn']
composite[my_list]
a = composite['actual_amount_paid']

b = composite['plan_list_price']

c = composite['payment_plan_days']

d = composite['is_churn']
plt.scatter(a,d)
plt.scatter(b,d)
plt.scatter(c,d)
plt.scatter(a,b)
plt.scatter(a,c)
plt.scatter(b,c)
payment_plan_days = composite[my_list].groupby('payment_plan_days').agg(['sum','count'])
payment_plan_days
payment_plan_days.groupby(axis=1, level=0).apply(lambda x: x.sum()/ x.count())
transactions
#pm_id = churn_rate(transactions, 'payment_method_id')
#pm_id.dtypes
#members = pd.merge(members, train, on='msno', how='left')

#members.shape