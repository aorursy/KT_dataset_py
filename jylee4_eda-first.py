import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

merchants = pd.read_csv('../input/merchants.csv')

new_merchant_t = pd.read_csv('../input/new_merchant_transactions.csv')

his_trans = pd.read_csv('../input/historical_transactions.csv')
print(train.shape)

print(test.shape)

print(merchants.shape)

print(new_merchant_t.shape)

print(his_trans.shape)
train.dtypes
train['first_active_month'] = pd.to_datetime(train['first_active_month']).apply(lambda x: x.strftime('%Y-%m'))
print(new_merchant_t.dtypes)

print(his_trans.dtypes)
new_merchant_t['city_id'] = new_merchant_t['city_id'].astype(object)

new_merchant_t['merchant_category_id'] = new_merchant_t['merchant_category_id'].astype(object)

new_merchant_t['category_2'] = new_merchant_t['category_2'].astype(object)

new_merchant_t['state_id'] = new_merchant_t['state_id'].astype(object)

new_merchant_t['subsector_id'] = new_merchant_t['subsector_id'].astype(object)

new_merchant_t['purchase_date'] = pd.to_datetime(new_merchant_t['purchase_date'])



his_trans['city_id'] = his_trans['city_id'].astype(object)

his_trans['merchant_category_id'] = his_trans['merchant_category_id'].astype(object)

his_trans['category_2'] = his_trans['category_2'].astype(object)

his_trans['state_id'] = his_trans['state_id'].astype(object)

his_trans['subsector_id'] = his_trans['subsector_id'].astype(object)

his_trans['purchase_date'] = pd.to_datetime(his_trans['purchase_date'])
print(merchants.dtypes)
merchants['merchant_group_id'] = merchants['merchant_group_id'].astype(object)

merchants['merchant_category_id'] = merchants['merchant_category_id'].astype(object)

merchants['subsector_id'] = merchants['subsector_id'].astype(object)

merchants['city_id'] = merchants['city_id'].astype(object)

merchants['state_id'] = merchants['state_id'].astype(object)

merchants['category_2'] = merchants['category_2'].astype(object)
train.head(10)
train['target'].describe()
f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(train.target)
f, ax = plt.subplots(figsize=(10, 8))

sns.distplot(train['target'])
# calculate the correlation matrix

corr = train.corr()



# plot the heatmap

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns, annot=True)
col_list = train.columns.tolist()

col_list = col_list[2:]

f = pd.melt(train, value_vars=col_list)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)

g = g.map(sns.distplot, "value")
col_list = test.columns.tolist()

col_list = col_list[2:]

f = pd.melt(test, value_vars=col_list)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)

g = g.map(sns.distplot, "value")
his_trans.head()
his_trans.describe()
new_merchant_t.describe()
fig, ax = plt.subplots(1, 4, figsize = (16, 6));

his_trans['authorized_flag'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='authorized_flag');

his_trans['category_1'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown', title='category_1');

his_trans['category_2'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='gold', title='category_2');

his_trans['category_3'].value_counts().sort_index().plot(kind='bar', ax=ax[3], color='purple', title='category_3');

plt.suptitle('Counts of categiories historical transaction');

fig, ax = plt.subplots(1, 4, figsize = (16, 6));

new_merchant_t['authorized_flag'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='authorized_flag');

new_merchant_t['category_1'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown', title='category_1');

new_merchant_t['category_2'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='gold', title='category_2');

new_merchant_t['category_3'].value_counts().sort_index().plot(kind='bar', ax=ax[3], color='purple', title='category_3');

plt.suptitle('Counts of categiories new merchant transaction');
his_trans[his_trans['authorized_flag']=='N'].head(10)
fig, ax = plt.subplots(1, 2, figsize = (16, 6));

his_trans['installments'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='installments');

his_trans['month_lag'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown', title='month_lag');

plt.suptitle('Counts of categiories historical transaction');



fig, ax = plt.subplots(1, 2, figsize = (16, 6));

new_merchant_t['installments'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='installments');

new_merchant_t['month_lag'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown', title='month_lag');

plt.suptitle('Counts of categiories new merchant transaction');
his_trans[his_trans['installments']==999].head()
his_trans[(his_trans['installments']==999) & (his_trans.authorized_flag=='Y')]
his_trans[his_trans['installments']==-1].head()
his_trans[his_trans['city_id']==-1]
his_trans[(his_trans['city_id']==-1)&(his_trans['state_id']!=-1)]
print(len(his_trans[(his_trans['city_id']!=-1)&(his_trans['state_id']==-1)]))

his_trans[(his_trans['city_id']!=-1)&(his_trans['state_id']==-1)].head()
merchants.isnull().sum()
merchants[merchants.avg_sales_lag3.isnull()==True]
new_merchant_t.isnull().sum()
his_trans.isnull().sum()
train = train[train.target>-30]
# purchase_amount size per card_id (구매건수)

c_his = his_trans.groupby("card_id")

c_his = c_his["purchase_amount"].size().reset_index()

c_his.columns = ["card_id","purchase_amount_size"]

train = pd.merge(train, c_his, on="card_id", how="left")
data = pd.concat([train['target'], train['purchase_amount_size']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='purchase_amount_size', y="target", data=data)

plt.xlabel("purchase_amount_size")

plt.ylabel("target")

plt.show
# purchase_amount mean per card_id

c_his = his_trans.groupby("card_id")

c_his = c_his["purchase_amount"].mean().reset_index()

c_his.columns = ["card_id","purchase_amount_mean"]

train = pd.merge(train, c_his, on="card_id", how="left")
data = pd.concat([train['target'], train['purchase_amount_mean']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='purchase_amount_mean', y="target", data=data)

plt.xlabel("purchase_amount_mean")

plt.ylabel("target")

plt.show
train[train['purchase_amount_mean']>400000]
his_trans[his_trans['purchase_amount']>400000]
# if remove that point,

data = data[data.purchase_amount_mean < 400000]

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='purchase_amount_mean', y="target", data=data)

plt.xlabel("purchase_amount_mean")

plt.ylabel("target")

plt.show
c_his = his_trans.groupby("card_id")

c_his = c_his["installments"].mean().reset_index()

c_his.columns = ["card_id","installments_mean"]

train = pd.merge(train, c_his, on="card_id", how="left")
data = pd.concat([train['target'], train['installments_mean']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='installments_mean', y="target", data=data)

plt.xlabel("installments_mean")

plt.ylabel("target")

plt.show
c_his = his_trans.groupby("card_id")

c_his = c_his["month_lag"].max().reset_index()

c_his.columns = ["card_id","month_lag_recent"]

train = pd.merge(train, c_his, on="card_id", how="left")
data = pd.concat([train['target'], train['month_lag_recent']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='month_lag_recent', y="target", data=data)

plt.xlabel("month_lag_recent")

plt.ylabel("target")

plt.show
his_trans['purchase_date_month'] = his_trans['purchase_date'].apply(lambda x: x.strftime('%Y-%m'))
c_his = his_trans.groupby("card_id")

c_his = c_his["purchase_date_month"].max().reset_index()

c_his.columns = ["card_id","purchase_date_month_recent"]

train = pd.merge(train, c_his, on="card_id", how="left")
train['purchase_period'] = pd.to_datetime(train['purchase_date_month_recent'])-pd.to_datetime(train['first_active_month'])
for i in range(len(train)):

    train.purchase_period[i] = train.purchase_period[i].days//30
# calculate the correlation matrix

corr = train.corr()



# plot the heatmap

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns, annot=True)
new_merchant_t.head()
# purchase_amount size per card_id (구매건수)

c_his = new_merchant_t.groupby("card_id")

c_his = c_his["purchase_amount"].size().reset_index()

c_his.columns = ["card_id","purchase_amount_size_new"]

train = pd.merge(train, c_his, on="card_id", how="left")
data = pd.concat([train['target'], train['purchase_amount_size_new']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='purchase_amount_size_new', y="target", data=data)

plt.xlabel("purchase_amount_size_new")

plt.ylabel("target")

plt.show
# purchase_amount mean per card_id

c_his = new_merchant_t.groupby("card_id")

c_his = c_his["purchase_amount"].mean().reset_index()

c_his.columns = ["card_id","purchase_amount_mean_new"]

train = pd.merge(train, c_his, on="card_id", how="left")
data = pd.concat([train['target'], train['purchase_amount_mean_new']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='purchase_amount_mean_new', y="target", data=data)

plt.xlabel("purchase_amount_mean_new")

plt.ylabel("target")

plt.show
c_his = new_merchant_t.groupby("card_id")

c_his = c_his["installments"].mean().reset_index()

c_his.columns = ["card_id","installments_mean_new"]

train = pd.merge(train, c_his, on="card_id", how="left")
data = pd.concat([train['target'], train['installments_mean_new']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='installments_mean_new', y="target", data=data)

plt.xlabel("installments_mean_new")

plt.ylabel("target")

plt.show
train[train.installments_mean_new>30]
new_merchant_t[new_merchant_t.installments==999]
# if remove that point,

data = data[data.installments_mean_new<30]

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='installments_mean_new', y="target", data=data)

plt.xlabel("installments_mean_new")

plt.ylabel("target")

plt.show
c_his = new_merchant_t.groupby("card_id")

c_his = c_his["month_lag"].max().reset_index()

c_his.columns = ["card_id","month_lag_recent_new"]

train = pd.merge(train, c_his, on="card_id", how="left")
data = pd.concat([train['target'], train['month_lag_recent_new']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='month_lag_recent_new', y="target", data=data)

plt.xlabel("month_lag_recent_new")

plt.ylabel("target")

plt.show
new_merchant_t['purchase_date_month'] = new_merchant_t['purchase_date'].apply(lambda x: x.strftime('%Y-%m'))
c_his = new_merchant_t.groupby("card_id")

c_his = c_his["purchase_date_month"].max().reset_index()

c_his.columns = ["card_id","purchase_date_month_recent_new"]

train = pd.merge(train, c_his, on="card_id", how="left")
train['purchase_period_new'] = pd.to_datetime(train['purchase_date_month_recent_new'])-pd.to_datetime(train['first_active_month'])
train.head()
# Fill null by most frequent data

#df_trans['category_2'].fillna(1.0,inplace=True)

#df_trans['category_3'].fillna('A',inplace=True)

#df_trans['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
train.isnull().sum()
train.dtypes