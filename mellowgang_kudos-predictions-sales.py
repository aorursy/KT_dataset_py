import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

import datetime

path ='../input'

print(os.listdir(path))
print('There are {} files in our folder'.format(len(os.listdir(path))))
%%time

train = pd.read_csv(os.path.join(path,'sales_train.csv'))
x,y = train.shape

print('There are {} rows and {} columns'.format(x,y))
print('Train columns {}'.format(train.columns))
train.info()
train.describe()
sns.set()

train.hist(bins=50,figsize=(15,8))#The bins parameter is used to custom the number of bins shown on the plots.

plt.show()
sub = pd.read_csv(os.path.join(path,'sample_submission.csv'))

test = pd.read_csv(os.path.join(path,'test.csv'))

test.head()
train.head()
corr_t = train.corr()

corr = corr_t.item_cnt_day.sort_values(ascending=False)

corr
sns.set()

plt.figure(figsize=(15,8))

sns.boxplot("item_cnt_day",data=train)
s= 0

for i in train.item_cnt_day:

    if(i>=1000):

        s+=1

        print('Outlier {} is {}'.format(s,i))
outlier1 = train['item_cnt_day']==1000

train[outlier1]
outlier2 = train['item_cnt_day']==2169

train[outlier2]
train.loc[train['item_cnt_day']>=1000,'item_cnt_day']= np.median(train['item_cnt_day'])
sns.set()

plt.figure(figsize=(15,8))

sns.boxplot('item_price',data=train)
out =train[train.item_price>300000]

out
train.loc[train.item_price>40000,'item_price']= np.median(train.item_price)
sns.set()

plt.figure(figsize=(15,8))

sns.boxplot('item_price',data=train)

plt.title('New boxplot item_price')
sns.set()

plt.figure(figsize=(15,8))

sns.boxplot("item_cnt_day",data=train)

plt.title('New boxplot item_cnt_day')
#Removing negative values too and replacing em' with median

train.loc[train.item_cnt_day<0,'item_cnt_day']=np.median(train.item_cnt_day)
sns.set()

plt.figure(figsize=(15,8))

sns.boxplot("item_cnt_day",data=train)

plt.title('New boxplot item_cnt_day')


train.date=train.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

train.head()
sns.set()

plt.figure(figsize=(15,8))

time_df = train.copy()

time_df.set_index('date',inplace=True)

time_df.item_cnt_day.plot()
month_sales =train.groupby(["date_block_num","shop_id","item_id"])[

    "date","item_price","item_cnt_day"].agg({"item_price":"mean","item_cnt_day":"sum"})



month_sales.head()