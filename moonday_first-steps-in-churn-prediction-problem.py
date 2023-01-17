import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

from datetime import datetime

train=pd.read_csv('../input/train.csv')

gr=train.groupby('is_churn').count()

gr.plot.pie(subplots=True,autopct='%.2f',figsize=(5, 5))
train.isnull().sum()
members=pd.read_csv('../input/members_v2.csv')

members.head()
members.isnull().sum()
members=members.fillna("NA")

members.groupby('gender').count().plot.bar(y='msno',figsize=(16, 5))
members.hist(figsize=(16, 10))
bin=list(range(-3200,1980, 100))

group=members.groupby(pd.cut(members.bd, bin)).count()

group.plot.bar(y='msno',figsize=(16, 5))
columns = ['gender','bd']

members=members.drop(columns, axis=1)
members['registration_init_time']=pd.to_datetime(members['registration_init_time'],format='%Y%m%d')
members.head()
transactions = pd.read_csv('../input/transactions.csv')

transactions.head()
transactions.isnull().sum()
transactions['transaction_date']=pd.to_datetime(transactions['transaction_date'],format='%Y%m%d') 

transactions['membership_expire_date']=pd.to_datetime(transactions['membership_expire_date'],format='%Y%m%d')
transactions.describe().transpose()
transactions.hist(column=['is_cancel', 'is_auto_renew'],figsize=(16, 5))

transactions.hist(column=['payment_method_id','payment_plan_days'],figsize=(16, 5),bins=50)
transactions.hist(column=['actual_amount_paid','plan_list_price'],figsize=(16, 5),bins=50)
transactions["actual_amount_paid"].median()
transactions["plan_list_price"].median()
logs = pd.read_csv('../input/user_logs.csv', nrows=30000000)
logs.head()
logs.isnull().sum()
logs['date']=pd.to_datetime(logs['date'],format='%Y%m%d') 
gr=logs.groupby(pd.Grouper(key='date', freq='D')).mean()
gr.plot.line(y=['num_25','num_50','num_75','num_985','num_100'],figsize=(16, 5))
gr.plot.line(y=['num_unq'],figsize=(16, 5))
train_total=pd.merge(transactions,pd.merge(members,train,on='msno',how='inner'),on='msno',how='inner')

train_total.head()
train_total.info()
gr=train_total.groupby(['transaction_date','is_churn']).count()
gr2=gr.unstack()

gr2.plot.line(y='msno',figsize=(16, 5))
gr2.plot.line(y='msno',figsize=(20, 8),ylim=(0,600),xlim=(pd.Timestamp('2015-10-01'), pd.Timestamp('2017-03-28')))
df1=train_total[train_total['is_cancel'] ==1]



gr3=df1.groupby(['transaction_date','is_churn']).sum()

gr4=gr3.unstack()

gr4.plot.line(y='is_cancel',figsize=(16, 5),ylim=(0,60),xlim=(pd.Timestamp('2015-10-01'), pd.Timestamp('2017-03-28')))
logs_total=pd.merge(logs,pd.merge(members,train,on='msno',how='inner'),on='msno',how='inner')

logs_total.head()
logs_total.info()
group1=logs_total.groupby(['date','is_churn']).sum()
group2=group1.unstack()

group2.plot.line(y='num_100',figsize=(16, 5))
group2.plot.line(y=['num_100'],figsize=(16, 5),ylim=(0,15000),xlim=(pd.Timestamp('2015-10-01'), pd.Timestamp('2017-03-28')))
group2.plot.line(y=['num_50'],figsize=(16, 5),ylim=(0,2000),xlim=(pd.Timestamp('2015-10-01'), pd.Timestamp('2017-03-28')))