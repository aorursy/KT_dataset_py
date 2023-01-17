import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime

from statsmodels.graphics.tsaplots import plot_acf

import seaborn as sns

import numpy as np

import statsmodels.api as sm

%matplotlib inline





#Reading data from the input directory

bitcoin_data = pd.read_csv('../input/bitcoin_dataset.csv', header=0, parse_dates=['Date'])

bitcoin_data['Year'] = bitcoin_data['Date'].apply(lambda x: x.year)

bitcoin_data['Month'] = bitcoin_data['Date'].apply(lambda x: x.month)

bitcoin_data.head(3)
plt.plot(bitcoin_data['Date'], bitcoin_data['btc_n_transactions_total'])

plt.show()
plt.plot(bitcoin_data['Date'], bitcoin_data['btc_hash_rate'])

plt.show()
bitcoin_data = bitcoin_data.loc[bitcoin_data['Date'] > datetime(2015,1,1)]



# Transform the total number of transactions into a scale of million

bitcoin_data['btc_n_transactions_total'] = bitcoin_data['btc_n_transactions_total']/1000000



# The dataset has btc_miners_revenue which is basically total value in bitcoin earned by miners. 

# However, from user perspective who wants to transact bitcoin, we should define another parameter. This

# parameter should provide a measure of average cost incurred by the user per transaction

bitcoin_data['Avg_Txn_Fee'] = bitcoin_data['btc_transaction_fees']/bitcoin_data['btc_n_transactions_total']

sns.pairplot(bitcoin_data[bitcoin_data.columns[[8,9,10,11,13,24]]],hue='Year',palette='afmhot')
# Median Txn time vs. Log(no. of transactions per block)

sns.lmplot('btc_n_transactions_per_block','btc_median_confirmation_time',

           data= pd.concat([bitcoin_data['btc_median_confirmation_time'],

            np.log(bitcoin_data['btc_n_transactions_per_block']),

            bitcoin_data['Year']],axis=1),hue='Year',fit_reg=False)



plt.xlabel('Log(No. of transactions/block)')

plt.ylabel('Median Time')
# Median Txn time vs Avg fee per transaction

sns.lmplot('Avg_Txn_Fee','btc_median_confirmation_time',

           data= pd.concat([bitcoin_data['btc_median_confirmation_time'],

            bitcoin_data['Year'], bitcoin_data['Avg_Txn_Fee']], axis=1),hue='Year',fit_reg=False)



plt.xlabel('Average Transaction fees')

plt.ylabel('Median Time')
bitcoin_data_2015 = bitcoin_data.loc[bitcoin_data['Year']==2015]

bitcoin_data_2016 = bitcoin_data.loc[bitcoin_data['Year']==2016]

bitcoin_data_2017 = bitcoin_data.loc[bitcoin_data['Year']==2017]



bitcoin_data_2016 = bitcoin_data_2016.loc[bitcoin_data_2016['btc_median_confirmation_time'] < 25]

bitcoin_data_2016 = bitcoin_data_2016.loc[bitcoin_data_2016['Avg_Txn_Fee'] < 2.5]

# Lets check the correlation between btc_n_transactions_per_block and Avg_Txn_Fee

print(np.corrcoef(bitcoin_data.loc[bitcoin_data['Year'] == 2015,'btc_n_transactions_per_block'],

            bitcoin_data.loc[bitcoin_data['Year'] == 2015, 'Avg_Txn_Fee'])[0][1])



print(np.corrcoef(bitcoin_data.loc[bitcoin_data['Year'] == 2016,'btc_n_transactions_per_block'],

            bitcoin_data.loc[bitcoin_data['Year'] == 2016, 'Avg_Txn_Fee'])[0][1])



print(np.corrcoef(bitcoin_data.loc[bitcoin_data['Year'] == 2017,'btc_n_transactions_per_block'],

            bitcoin_data.loc[bitcoin_data['Year'] == 2017, 'Avg_Txn_Fee'])[0][1])
# Regression year 2015

# Median confirmation time ~ log(no. of transactions/block) + average transaction fee

reg_data_2015 = bitcoin_data_2015[['btc_median_confirmation_time', 'btc_n_transactions_per_block', 

                                   'Avg_Txn_Fee']]

reg_data_2015['log_txn_block'] = reg_data_2015['btc_n_transactions_per_block'].apply(lambda x: np.log(x))

reg_data_2015 = reg_data_2015.drop('btc_n_transactions_per_block', axis=1)

reg_data_2015_exog = sm.add_constant(reg_data_2015[['Avg_Txn_Fee', 'log_txn_block']], prepend=False)

model_2015 = sm.OLS(reg_data_2015['btc_median_confirmation_time'],reg_data_2015_exog)

model_2015.fit().summary()
# Regression year 2016

# Median confirmation time ~ log(no. of transactions/block) + average transaction fee

reg_data_2016 = bitcoin_data_2016[['btc_median_confirmation_time', 'btc_n_transactions_per_block', 'Avg_Txn_Fee']]

reg_data_2016['log_txn_block'] = reg_data_2016['btc_n_transactions_per_block'].apply(lambda x: np.log(x))

reg_data_2016 = reg_data_2016.drop('btc_n_transactions_per_block', axis=1)

reg_data_2016_exog = sm.add_constant(reg_data_2016[['Avg_Txn_Fee', 'log_txn_block']], prepend=False)

model_2016 = sm.OLS(reg_data_2016['btc_median_confirmation_time'],reg_data_2016_exog)

model_2016.fit().summary()
# Regression year 2017

# Median confirmation time ~ log(no. of transactions/block) + average transaction fee

reg_data_2017 = bitcoin_data_2017[['btc_median_confirmation_time', 'btc_n_transactions_per_block', 'Avg_Txn_Fee']]

reg_data_2017['log_txn_block'] = reg_data_2017['btc_n_transactions_per_block'].apply(lambda x: np.log(x))

reg_data_2017 = reg_data_2017.drop('btc_n_transactions_per_block', axis=1)

reg_data_2017_exog = sm.add_constant(reg_data_2017[['Avg_Txn_Fee', 'log_txn_block']], prepend=False)

model_2017 = sm.OLS(reg_data_2017['btc_median_confirmation_time'],reg_data_2017_exog)

model_2017.fit().summary()
# Time Series analysis of log of avg. no transactions per block over time.

plt.plot(bitcoin_data['Date'], np.log(bitcoin_data['btc_n_transactions_per_block']))

plt.xticks(rotation=45)

plt.xlabel('Date')

plt.ylabel('Log(Avg_n_transactions)')
block_txn = pd.DataFrame(bitcoin_data.loc[bitcoin_data['Date'] < datetime(2017,7,31), 'btc_n_transactions_per_block'])

block_txn['log_txn_block'] = np.log(block_txn[['btc_n_transactions_per_block']])

block_txn = block_txn.drop('btc_n_transactions_per_block', axis=1)



block_txn['time'] = block_txn.index - block_txn.index[0] + 1

block_txn_exog = sm.add_constant(block_txn[['time']], prepend=False)

model_txn_blk = sm.OLS(block_txn['log_txn_block'], block_txn_exog)

results_txn_blk = model_txn_blk.fit()



plot_acf(results_txn_blk.resid, lags=50)
block_txn['log_txn_block_lag_7'] = block_txn['log_txn_block'].shift(7)

block_txn['log_txn_block_lag_1'] = block_txn['log_txn_block'].shift(1)



block_txn = block_txn.dropna()

model_txn_blk = sm.OLS(block_txn['log_txn_block'], block_txn[['log_txn_block_lag_7', 'log_txn_block_lag_1']])

results_txn_blk = model_txn_blk.fit()

plot_acf(results_txn_blk.resid, lags=50)