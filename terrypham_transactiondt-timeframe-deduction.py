%load_ext autoreload



%autoreload 2



from IPython.display import display

import pandas as pd

pd.options.display.max_columns = None

from IPython.display import display, HTML





import datetime

from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

import xgboost as xgb

from statsmodels.tsa.seasonal import seasonal_decompose

import seaborn as sns

train_trans = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
test_trans = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
plt.figure(figsize=(10,5))

_ = plt.hist(train_trans['TransactionDT'], bins=100), plt.hist(test_trans['TransactionDT'], bins=100)

plt.legend(['train','test'])
temp = train_trans.merge(train_id,on='TransactionID',how='inner')
temp.groupby('DeviceInfo').agg({'TransactionDT':'min'}).sort_values('TransactionDT').head(20)
np.max(test_trans['TransactionDT'])/86400 - np.min(train_trans['TransactionDT'])/86400
trends = pd.read_csv('../input/google-trends-shopping-data/multiTimeline.csv')
plt.figure(figsize=(20,10))

plt.xticks(rotation=65)

_ = plt.plot('Week','Traffic',data=trends)
plt.figure(figsize=(20,10))

train, test = plt.hist(np.ceil(train_trans['TransactionDT']/86400), bins=182), plt.hist(np.ceil(test_trans['TransactionDT']/86400), bins=182)

train[1][:182][train[0]> 6000]
test_peaks = test[1][:182][test[0]> 5000]
test_peaks
[datetime.date(2017,11,1) + datetime.timedelta(days=x) for x in test_peaks.tolist()]