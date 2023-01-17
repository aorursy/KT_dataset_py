import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import datetime

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
# Loading transaction data

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')



train_transaction['TransactionDT'] = train_transaction['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

train_transaction['trans_yymm'] = train_transaction['TransactionDT'].map(lambda x: x.replace(day=1))

train_transaction['trans_date'] = train_transaction['TransactionDT'].map(lambda x: x.date())
fraud_rate_daily = train_transaction.groupby(['trans_date'])['isFraud'].mean()



#Determing 30 day rolling statistics

rolmean = fraud_rate_daily.rolling(30).mean()

rolstd = fraud_rate_daily.rolling(30).std()



#Plot rolling statistics:

orig = plt.plot(fraud_rate_daily, color='blue',label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label = 'Rolling Std')

plt.legend(loc='best')

plt.title('Fraud Rate: Rolling Mean & Standard Deviation')

plt.show(block=False)

num_frauds_daily = train_transaction.groupby(['trans_date'])['isFraud'].sum()



#Determing 30 day rolling statistics

rolmean = num_frauds_daily.rolling(30).mean()

rolstd = num_frauds_daily.rolling(30).std()



#Plot rolling statistics:

orig = plt.plot(num_frauds_daily, color='blue',label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label = 'Rolling Std')

plt.legend(loc='best')

plt.title('# Fraud transactions: Rolling Mean & Standard Deviation')

plt.show(block=False)
# Fraud rate

dftest = adfuller(fraud_rate_daily, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
# Number of Fraud transactions

dftest = adfuller(num_frauds_daily, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)