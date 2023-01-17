import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(rc={'figure.figsize':(6,5)});
plt.figure(figsize=(6,5));

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
print(data.shape)
data.head()
# Checking inbalance
data.isFraud.value_counts()
p = sns.countplot(data=data, x='isFraud')
#p.set(ylim = (-10000, 500000))

plt.ylabel('Count')
data.isFraud.value_counts(normalize=True)*100
data[data.step > 718].isFraud.value_counts()
data = data[data.step <= 718]
print(data.shape)
# Checking inbalance
data.isFraud.value_counts()
data.type.value_counts()
# Fraud occurs only among 2 type of transactions
data.groupby('type')['isFraud'].sum()
data.groupby('type')['isFlaggedFraud'].sum()
data[data.isFlaggedFraud == 1]
# Missing Values
data.isnull().values.any()
data[(data.step > 50) & (data.step < 90)].isFraud.value_counts()
data[data.isFraud == 1].shape
data[data.isFraud == 1].nameDest.value_counts()
data[data.nameDest == 'C1259079602']
data.describe()
data[data.amount > 1500000].shape
p = sns.boxplot(data=data, x='isFraud', y='amount')
p = sns.boxplot(data=data, x='isFraud', y='amount')
p.set(ylim = (0, 4000000))
sns.boxplot(data=data[data.amount < 1500000], x='isFraud', y='amount')
data[data.isFraud == 0].amount.mean()
data[data.isFraud == 0].amount.describe()
data[data.isFraud == 1].amount.describe()
data[(data.amount < 1) & (data.isFraud == 0)]
data[(data.amount == 0) & (data.isFraud == 1)]
set(data[data.isFraud == 1].nameOrig).intersection(set(data[data.isFraud == 0].nameDest.unique()))
data[data.isFraud == 0].nameDest.sort_values()
data[data.isFraud == 1].nameOrig.sort_values()
data[data.nameOrig == 'C1510987794']
p = sns.boxplot(data=data, x='isFraud', y='oldbalanceOrg')
p.set(ylim = (0, 4000000))
plt.ylabel('Opening Balance')
p = sns.boxplot(data=data, x='isFraud', y='newbalanceOrig')
p = sns.boxplot(data=data, x='isFraud', y='newbalanceOrig')
#p.set(ylim = (-10000, 500000))
p.set(ylim = (-100000, 4000000))
plt.ylabel('Origin Closing Balance')
data[data.isFraud == 1].oldbalanceOrg.describe()
p = sns.boxplot(data=data, x='isFraud', y='oldbalanceDest')
p.set(ylim = (-10000, 1500000))
plt.ylabel('Destination Opening Balance')
p = sns.boxplot(data=data, x='isFraud', y='newbalanceDest')
p.set(ylim = (-100000, 1500000))
plt.ylabel('Destination Closing Balance')
data[data.isFraud == 1].amount.hist(bins=30)
data.head()
data['amount'] = np.log1p(data['amount'])
data['oldbalanceOrg'] = np.log1p(data['oldbalanceOrg'])
data['newbalanceOrig'] = np.log1p(data['newbalanceOrig'])
data['oldbalanceDest'] = np.log1p(data['oldbalanceDest'])
data['newbalanceDest'] = np.log1p(data['newbalanceDest'])
data.head()
p = sns.boxplot(data=data, x='isFraud', y='amount')
p = sns.boxplot(data=data, x='isFraud', y='oldbalanceOrg')
plt.ylabel('Opening Balance')
p = sns.boxplot(data=data, x='isFraud', y='newbalanceOrig')
plt.ylabel('Origin Closing Balance')
data.head()
p = sns.boxplot(data=data, x='isFraud', y='oldbalanceDest')
plt.ylabel('Destination Opening Balance')
p = sns.boxplot(data=data, x='isFraud', y='newbalanceDest')
plt.ylabel('Destination Closing Balance')



data.step.value_counts()
data.tail()
data[(data.step > 700) & (data.step < 710)].isFraud.value_counts()
data[(data.step == 718)].isFraud.value_counts()
718/24
