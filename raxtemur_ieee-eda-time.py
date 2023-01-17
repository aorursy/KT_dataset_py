import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import datetime

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
print(train.shape, test.shape)
del train_transaction, train_identity, test_transaction, test_identity

gc.collect()
train.head()
df = pd.concat([train, test], axis = 0, sort = False)
del train, test

gc.collect()
START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')



date = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)).timetuple())
time = pd.DataFrame(data={'Day': date.apply(lambda x: x[2]), 'Sec': date.apply(lambda x: 3600*x[3]+60*x[4]+x[5]), 'Wday': date.apply(lambda x: x[6])})

time
del date

gc.collect()
time = time.merge(df['isFraud'], how='left', left_index=True, right_index=True)

time
time['Hour'] = time['Sec']//3600

time[:590540].describe()
plt.plot(time[:590540].groupby('Hour').mean()['isFraud'], color='k')

ax = plt.gca()

ax2 = ax.twinx()

ax2.hist(time[:590540]['Hour'], alpha=0.5, bins=24)



ax.set_ylabel('Fraction of fraudulent transactions')

ax2.set_ylabel('Number of transactions')
plt.plot(time[:590540].groupby('Wday').mean()['isFraud'], color='k')

ax = plt.gca()

ax2 = ax.twinx()

ax2.hist(time[:590540]['Wday'], alpha=0.5, bins=7)



ax.set_ylabel('Fraction of fraudulent transactions')

ax2.set_ylabel('Number of transactions')
%%time

missing_values_count = df.isnull().sum()

missing_values_destribution = [np.sum(missing_values_count/1097231 > 0.01*x) for x in range(0, 100)]
plt.bar(range(100), missing_values_destribution, color='b', alpha=0.7)
s = (missing_values_count/1097231 > 0.79)

NaNindexes = s[s].index

NaNindexes
df = df.drop(NaNindexes, axis=1)

df.isna().sum()
df.to_csv("df.csv", index=True)

time.to_csv("time.csv", index=True)