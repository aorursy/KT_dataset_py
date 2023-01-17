import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train = pd.read_csv('../input/train_2016.csv')

train.shape
train.head()
prop = pd.read_csv('../input/properties_2016.csv')
prop.shape
prop.head()
train_merge= pd.merge(train,prop,on='parcelid',how='left')
train_merge.shape
train_merge.head()
count = train_merge.isnull().sum().sort_values(ascending= False)

percent = count/train_merge.shape[0]

missing_data = pd.concat([count,percent],keys=['count','percent'],axis=1)

missing_data
train_merge.columns
train_merge.dtypes
num_features = train_merge.select_dtypes(include=[np.number]).columns

cat_features = train_merge.select_dtypes(exclude=[np.number]).columns
cat_features
num_features.shape[0]
# number of columns having missing values

missing_data[missing_data['count']>0].index.shape[0]
# Is categorical features has missing values

train_merge[cat_features].isnull().sum().sort_values(ascending = False)
sns.distplot(train_merge['logerror'], bins=50)
train_merge['logerror'].skew()
train_merge['logerror'].describe()
train_merge['transactiondate'].value_counts()
train_merge_ = train_merge
train_merge_['transactiondate'] = pd.to_datetime(train_merge_['transactiondate'])

train_merge_['transaction_month'] = train_merge_['transactiondate'].dt.month
train_merge_[['transaction_month', 'transactiondate']].head()
train_merge_['transaction_month'].value_counts()
#Let us first check the number of transactions in each month



cnt_srs = train_merge_['transaction_month'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values)

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
#Let us first check the number of transactions in each day

train_merge_['transaction_day'] = train_merge_['transactiondate'].dt.day



cnt_srs = train_merge_['transaction_day'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values)

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
corr = train_merge_[num_features].corr()

sns.heatmap(corr)
# Let's check the first ten features are the most positively correlated with SalePrice and 

# the next ten are the most negatively correlated.



print (corr['logerror'].sort_values(ascending=False)[:10], '\n') #top 10 values

print ('----------------------')

print (corr['logerror'].sort_values(ascending=False)[-10:]) #last 10 values`
train_merge_['fireplaceflag'].unique()
train_merge_['fireplaceflag']= train_merge_['fireplaceflag'].fillna('False')
train_merge_['taxdelinquencyflag'].unique()
train_merge_['taxdelinquencyflag'] = train_merge_['taxdelinquencyflag'].fillna('N')
train_merge_['hashottuborspa'].unique()
train_merge_['hashottuborspa'] = train_merge_['hashottuborspa'].fillna('False')
len(train_merge_['propertyzoningdesc'].unique())
train_merge_['propertycountylandusecode'].unique()