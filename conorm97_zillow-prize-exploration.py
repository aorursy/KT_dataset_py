import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
#load training file

train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

print(train.head())

print('---------------------')

print(train.shape)
log_errors = train['logerror']

upper_lim = np.percentile(log_errors, 99.5)

lower_lim = np.percentile(log_errors, 0.5)

log_errors = log_errors.clip(lower=lower_lim, upper=upper_lim)





plt.figure(figsize=(12,10))

plt.hist(log_errors, bins=300)

plt.title('Distribution of Target Variable (log-error)')

plt.ylabel('count')

plt.xlabel('log-error')

plt.show()
#load property features/description file

prop = pd.read_csv("../input/properties_2016.csv")

print(prop.head())

print('---------------------')

print(prop.shape)
nans = prop.drop('parcelid', axis=1).isnull().sum()

nans.sort_values(ascending=True, inplace=True)

nans = nans / prop.shape[0]

#print(nans)
plt.figure(figsize=(14, 5))

plt.bar(range(len(nans.index)), nans.values)

plt.xticks(range(len(nans.index)), nans.index.values, rotation=90)

plt.show()
train['transaction_month'] = pd.DatetimeIndex(train['transactiondate']).month

train.sort_values('transaction_month', axis=0, ascending=True, inplace=True)

print(train.head())
ax = sns.stripplot(x=train['transaction_month'], y=train['logerror'])
ax1 = sns.stripplot(x=train['transaction_month'][train['transaction_month'] > 9], y=train['logerror'])
trans = train['transaction_month'].value_counts(normalize=True)

trans = pd.DataFrame(trans)

trans['month'] = trans.index

trans = trans.sort_values('month', ascending=True)

trans.set_index('month')

trans.rename({'transaction_month' : ''})

print(trans)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.figure(figsize=(12, 5))

plt.bar(range(len(months)), trans['transaction_month'])

plt.title('Proportion of Transactions per Month')

plt.ylabel('Proportion')

plt.xlabel('Month')

plt.xticks(range(len(months)), months, rotation=90)

plt.show()
#fill NaN values with -1 and encode object columns 

for x in prop.columns:

    prop[x] = prop[x].fillna(-1)
#many more parcelids in properties file, merge with training file

train = pd.merge(train, prop, on='parcelid', how='left')

print(train.head())

print('---------------------')

print(train.shape)

for c in train[['transactiondate', 'hashottuborspa', 'propertycountylandusecode', 'propertyzoningdesc', 'fireplaceflag', 'taxdelinquencyflag']]:

    label = LabelEncoder()

    label.fit(list(train[c].values))

    train[c] = label.transform(list(train[c].values))
x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)

y_train = train['logerror']



print(x_train.head())

print('------------')

print(y_train.head())
rf = RandomForestRegressor(n_estimators=30, max_features=None)
rf.fit(x_train, y_train)
rf_importance = rf.feature_importances_





importance = pd.DataFrame()

importance['features'] = x_train.columns

importance['importance'] = rf_importance

print(importance.head())

importance.sort_values('importance', axis=0, inplace=True, ascending=False)

print('------------')

print(importance.head())
fig = plt.figure(figsize=(10, 4), dpi=100)

plt.bar(range(len(importance)), importance['importance'])

plt.title('Feature Importances')

plt.xlabel('Feature Name')

plt.ylabel('Importance')

plt.xticks(range(len(importance)), importance['features'], rotation=90)

plt.show()