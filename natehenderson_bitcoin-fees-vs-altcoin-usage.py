import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Generated files are sent to "../input/bitcoin-blockchain/"

import os

print(os.listdir("../input"))
# Tools to help with Google BigQuery



from google.cloud import bigquery

from bq_helper import BigQueryHelper



client = bigquery.Client()



# Variable for each blockchain queried



btc_bq_assistant = BigQueryHelper("bigquery-public-data", "crypto_bitcoin")

bch_bq_assistant = BigQueryHelper("bigquery-public-data", "crypto_bitcoin_cash")

ltc_bq_assistant = BigQueryHelper("bigquery-public-data", "crypto_litecoin")

dash_bq_assistant = BigQueryHelper("bigquery-public-data", "crypto_dash")

doge_bq_assistant = BigQueryHelper("bigquery-public-data", "crypto_dogecoin")

eth_bq_assistant = BigQueryHelper("bigquery-public-data", "crypto_ethereum")

# Check Bitcoin (BTC) blockchain schema



btc_bq_assistant.list_tables()
# Google BigQuery on Kaggle has a free 5 TB limit

# These output total size of each blockchain's transaction history



btc_dataset = client.get_dataset(client.dataset('crypto_bitcoin', project='bigquery-public-data'))

btc_transactions_table = client.get_table(btc_dataset.table('transactions'))

BYTES_PER_GB = 2**30

print(f'The Bitcoin transactions table is {int(btc_transactions_table.num_bytes/BYTES_PER_GB)} GB')



bch_dataset = client.get_dataset(client.dataset('crypto_bitcoin_cash', project='bigquery-public-data'))

bch_transactions_table = client.get_table(bch_dataset.table('transactions'))

print(f'The Bitcoin Cash transactions table is {int(bch_transactions_table.num_bytes/BYTES_PER_GB)} GB')



ltc_dataset = client.get_dataset(client.dataset('crypto_litecoin', project='bigquery-public-data'))

ltc_transactions_table = client.get_table(ltc_dataset.table('transactions'))

print(f'The Litecoin transactions table is {int(ltc_transactions_table.num_bytes/BYTES_PER_GB)} GB')



dash_dataset = client.get_dataset(client.dataset('crypto_dash', project='bigquery-public-data'))

dash_transactions_table = client.get_table(dash_dataset.table('transactions'))

print(f'The Dash transactions table is {int(dash_transactions_table.num_bytes/BYTES_PER_GB)} GB')



doge_dataset = client.get_dataset(client.dataset('crypto_dogecoin', project='bigquery-public-data'))

doge_transactions_table = client.get_table(doge_dataset.table('transactions'))

print(f'The Dogecoin transactions table is {int(doge_transactions_table.num_bytes/BYTES_PER_GB)} GB')



eth_dataset = client.get_dataset(client.dataset('crypto_ethereum', project='bigquery-public-data'))

eth_transactions_table = client.get_table(eth_dataset.table('transactions'))

print(f'The Ethereum transactions table is {int(eth_transactions_table.num_bytes/BYTES_PER_GB)} GB')

# Anatomy of a Bitcoin transaction



btc_transactions_header = client.list_rows(btc_transactions_table, max_results=1)

print('\n'.join([str(dict(i)) for i in btc_transactions_header]))
# Create function to estimate query size before executing

# Mindful of 5 TB monthly limit on Google BigQuery data



def estimate_gigabytes_scanned(query, bq_client):

    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun

    my_job_config = bigquery.job.QueryJobConfig()

    my_job_config.dry_run = True

    my_job = client.query(query, job_config=my_job_config)

    BYTES_PER_GB = 2**30

    return my_job.total_bytes_processed / BYTES_PER_GB
# Query pulls BTC fees (independent variable of interest) by day



QUERY = """ 

                 SELECT DATE(block_timestamp) AS trans_date,

                        (SUM(input_value - output_value))/100000000 AS BTC_fees,

                        SUM(output_value)/100000000 AS BTC_transferred

                        

                 FROM `bigquery-public-data.crypto_bitcoin.transactions`

                 

                 /* Data from 2 months on either side of prediction, for comparison */

                 

                 WHERE DATE(block_timestamp) BETWEEN '2019-01-29' AND '2019-06-28'

                 GROUP BY trans_date

                 ORDER BY trans_date

                 """

estimate_gigabytes_scanned(QUERY, client)
# Set query limit at 100 GB to stay under limit



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**11)

query_job = client.query(QUERY, job_config=safe_config)



btc_data = query_job.to_dataframe()

btc_data.tail()
QUERY = """ 

                 SELECT DATE(block_timestamp) AS trans_date,

                        SUM(output_value)/100000000 AS BCH_transferred

                        

                 FROM `bigquery-public-data.crypto_bitcoin_cash.transactions`

                 WHERE DATE(block_timestamp) BETWEEN '2019-01-29' AND '2019-05-28'

                 GROUP BY trans_date

                 ORDER BY trans_date

                 """

estimate_gigabytes_scanned(QUERY, client)
# Set query limit at 100 GB



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**11)

query_job = client.query(QUERY, job_config=safe_config)



bch_data = query_job.to_dataframe()

bch_data.head()
QUERY = """ 

                 SELECT DATE(block_timestamp) AS trans_date,

                        SUM(output_value)/100000000 AS LTC_transferred

                        

                 FROM `bigquery-public-data.crypto_litecoin.transactions`

                 WHERE DATE(block_timestamp) BETWEEN '2019-01-29' AND '2019-05-28'

                 GROUP BY trans_date

                 ORDER BY trans_date

                 """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**12)

query_job = client.query(QUERY, job_config=safe_config)



ltc_data = query_job.to_dataframe()

ltc_data.head()
QUERY = """ 

                 SELECT DATE(block_timestamp) AS trans_date,

                        SUM(output_value)/100000000 AS DASH_transferred

                        

                 FROM `bigquery-public-data.crypto_dash.transactions`

                 WHERE DATE(block_timestamp) BETWEEN '2019-01-29' AND '2019-05-28'

                 GROUP BY trans_date

                 ORDER BY trans_date

                 """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**12)

query_job = client.query(QUERY, job_config=safe_config)



dash_data = query_job.to_dataframe()

dash_data.head()
QUERY = """ 

                 SELECT DATE(block_timestamp) AS trans_date,

                        SUM(output_value)/100000000 AS DOGE_transferred

                        

                 FROM `bigquery-public-data.crypto_dogecoin.transactions`

                 WHERE DATE(block_timestamp) BETWEEN '2019-01-29' AND '2019-05-28'

                 GROUP BY trans_date

                 ORDER BY trans_date

                 """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**12)

query_job = client.query(QUERY, job_config=safe_config)



doge_data = query_job.to_dataframe()

doge_data.head()

eth_transactions_header = client.list_rows(eth_transactions_table, max_results=1)

print('\n'.join([str(dict(i)) for i in eth_transactions_header]))
QUERY = """ 

                 SELECT DATE(block_timestamp) AS trans_date,

                        (SUM(value))/1000000000000000000 AS ETH_transferred

                        

                 FROM `bigquery-public-data.crypto_ethereum.transactions`

                 WHERE DATE(block_timestamp) BETWEEN '2019-01-29' AND '2019-05-28'

                 GROUP BY trans_date

                 ORDER BY trans_date

                 """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**12)

query_job = client.query(QUERY, job_config=safe_config)



eth_data = query_job.to_dataframe()

eth_data.head()
# Merge all dataframes



combined_data = pd.merge(btc_data,bch_data,on="trans_date")

combined_data = pd.merge(combined_data,ltc_data,on="trans_date")

combined_data = pd.merge(combined_data,dash_data,on="trans_date")

combined_data = pd.merge(combined_data,doge_data,on="trans_date")

combined_data = pd.merge(combined_data,eth_data,on="trans_date")

combined_data.head()
combined_data.tail()
# See file directory

!ls ../input/cc-prices
# Historical data imported into Google Sheets from CoinMarketCap using ImportHTML function.

# Example URL: https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20190129&end=20190629)

# Sheet created for each currency, same time frame plus one month of test data



btc_historical = pd.read_csv('../input/cc-prices/CC Prices - BTC.csv')

bch_historical = pd.read_csv('../input/cc-prices/CC Prices - BCH.csv')

ltc_historical = pd.read_csv('../input/cc-prices/CC Prices - LTC.csv')

dash_historical = pd.read_csv('../input/cc-prices/CC Prices - DASH.csv')

doge_historical = pd.read_csv('../input/cc-prices/CC Prices - DOGE.csv')

eth_historical = pd.read_csv('../input/cc-prices/CC Prices - ETH.csv')
# Removes commas

btc_historical["Open*"] = btc_historical["Open*"].replace({',':''}, regex=True)

# Changes data types from string to int

btc_historical["Open*"] = pd.to_numeric(btc_historical["Open*"])

# Removes commas from Close** column

btc_historical["Close**"] = btc_historical["Close**"].replace({',':''}, regex=True)

# Changes data types from string to int

btc_historical["Close**"] = pd.to_numeric(btc_historical["Close**"])

# Creates new column "Price" using the average of Open and Close

btc_historical["Price"] = (btc_historical["Open*"]+btc_historical["Close**"])/2
btc_historical.head()
# Reindexes data in chronological order

# Assigns to new column in combined_data

combined_data["BTC_price"] = btc_historical["Price"].iloc[::-1].reset_index(drop=True)

combined_data["BTC_price"].head()
# Creates new column in combined_data using the median daily price for each altcoin



combined_data["BCH_price"] = (bch_historical["Open*"]+bch_historical["Close**"])/2

combined_data["LTC_price"] = (ltc_historical["Open*"]+ltc_historical["Close**"])/2

combined_data["DASH_price"] = (dash_historical["Open*"]+dash_historical["Close**"])/2

combined_data["DOGE_price"] = (doge_historical["Open*"]+doge_historical["Close**"])/2

combined_data["ETH_price"] = (eth_historical["Open*"]+eth_historical["Close**"])/2



# Reindexes new column to chronological order



combined_data["BCH_price"] = combined_data["BCH_price"].iloc[::-1].reset_index(drop=True)

combined_data["LTC_price"] = combined_data["LTC_price"].iloc[::-1].reset_index(drop=True)

combined_data["DASH_price"] = combined_data["DASH_price"].iloc[::-1].reset_index(drop=True)

combined_data["DOGE_price"] = combined_data["DOGE_price"].iloc[::-1].reset_index(drop=True)

combined_data["ETH_price"] = combined_data["ETH_price"].iloc[::-1].reset_index(drop=True)
combined_data.head()
# Create new columns for amount of USD transferred by coin

# Make columns compatible with .astype('float')

# Convert amounts to $ millions



combined_data["USD_BTC_transferred"] = combined_data["BTC_transferred"].astype('float') * combined_data["BTC_price"] / 1000000

combined_data["USD_BCH_transferred"] = combined_data["BCH_transferred"].astype('float') * combined_data["BCH_price"] / 1000000

combined_data["USD_LTC_transferred"] = combined_data["LTC_transferred"].astype('float') * combined_data["LTC_price"] / 1000000

combined_data["USD_DASH_transferred"] = combined_data["DASH_transferred"].astype('float') * combined_data["DASH_price"] / 1000000

combined_data["USD_DOGE_transferred"] = combined_data["DOGE_transferred"].astype('float') * combined_data["DOGE_price"] / 1000000

combined_data["USD_ETH_transferred"] = combined_data["ETH_transferred"].astype('float') * combined_data["ETH_price"] / 1000000



combined_data.head()
# Show altcoin usage relative to BTC fees



import matplotlib

import matplotlib.pyplot as plt

from matplotlib.dates import date2num



date = combined_data["trans_date"]

fees = combined_data["BTC_fees"]

usdbtc = combined_data["USD_BTC_transferred"]

usdbch = combined_data["USD_BCH_transferred"]

usdltc = combined_data["USD_LTC_transferred"]

usddash = combined_data["USD_DASH_transferred"]

usddoge = combined_data["USD_DOGE_transferred"]

usdeth = combined_data["USD_ETH_transferred"]



fig, ax1 = plt.subplots(figsize=(20,10))



ax2 = ax1.twinx()

ax1.plot(date, usdbch)

ax1.plot(date, usdltc)

ax1.plot(date, usddash)

ax1.plot(date, usddoge)

ax1.plot(date, usdeth)

ax2.bar(date.apply(date2num), fees, color=(0.2, 0.4, 0.6, 0.15))



ax1.set_ylabel('USD ($) in millions',size=20, rotation = 90)

ax2.set_ylabel('Bitcoin (BTC) paid in fees',size=20)

ax1.legend(loc='upper left',prop={'size': 20})

plt.title(label='Value Transferred By Altcoin vs BTC Fees',size=30)



# Tilt x-axis labels

for tick in ax1.get_xticklabels():

    tick.set_rotation(315)



plt.show()

# Pearson correlation between BTC fees and value transferred of each altcoin



from scipy.stats import pearsonr



fees = combined_data["BTC_fees"].astype('float')

bch = combined_data["USD_BCH_transferred"].astype('float')

ltc = combined_data["USD_LTC_transferred"].astype('float')

dash = combined_data["USD_DASH_transferred"].astype('float')

doge = combined_data["USD_DOGE_transferred"].astype('float')

eth = combined_data["USD_ETH_transferred"].astype('float')



print("BCH", pearsonr(fees, bch))

print("LTC", pearsonr(fees, ltc))

print("DASH", pearsonr(fees, dash))

print("DOGE", pearsonr(fees, doge))

print("ETH", pearsonr(fees, eth))

# Our independent variables of interest is BTC fees

# Independent variable is ETH price the next day

# We used closing price to simulate at least a 24 hour window



df = pd.DataFrame()

df["fees"] = fees

df["ETH_price"] = eth_historical["Close**"].iloc[::-1].reset_index(drop=True)

df["ETH_price_next_day"] = df["ETH_price"].shift(-1)

df.drop([106],axis=0,inplace=True)

df



pearsonr(df["fees"],df["ETH_price_next_day"])
# LTC price shows the same level of correlation



df = pd.DataFrame()

df["fees"] = fees

df["LTC_price"] = btc_historical["Close**"].iloc[::-1].reset_index(drop=True)

df["LTC_price_next_day"] = df["LTC_price"].shift(-1)

df.drop([106],axis=0,inplace=True)

df



pearsonr(df["fees"],df["LTC_price_next_day"])
# The rate of increase may also affect ETH price

# E.g. a surge in fees could drive many more people to ETH



df = pd.DataFrame()

df["fees"] = fees

df["ETH_price"] = eth_historical["Close**"].iloc[::-1].reset_index(drop=True)

df["ETH_price_next_day"] = df["ETH_price"].shift(-1)

df.drop([106],axis=0,inplace=True)



df["delta_fees"] = df["fees"].shift(-1) - df["fees"]

df.drop([105],inplace=True)



pearsonr(df["delta_fees"],df["ETH_price_next_day"])
# Predict whether ETH will go up the next day with a logistic regression on Bitcoin fees

# If ETH_price goes up the next day (price_increase is greater than 0), we want to buy

# price_increase is a buy signal



df["price_increase"] = df["ETH_price_next_day"] - df["ETH_price"] >= 0
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report



cols = ["fees"]

X = df[cols]

y = df["price_increase"]



df_test = pd.DataFrame()

df_test["fees"] = btc_data["BTC_fees"]

df_test["ETH_price"] = eth_historical["Close**"].iloc[::-1].reset_index(drop=True)

df_test["BCH_price"] = bch_historical["Close**"].iloc[::-1].reset_index(drop=True)

# df_test["delta_fees"] = df_test["fees"].shift(-1) - df_test["fees"]

df_test = df_test[105::]

df_test.drop([150],inplace=True)



model = LogisticRegression()

model.fit(X,y)



X_test = df_test[cols]

y_test = df_test["ETH_price"] - df_test["ETH_price"].shift(-1)

y_test.iloc[-1] = True

y_pred = model.predict(X_test)



y_test = y_test > 0
print("The model suggested buying ETH",sum(y_pred),"of",y_pred.size,"days.")
model.score(X_test,y_test)
# Start with $100,000

# Buy and sell ETH when model says



ETH = 0

bank = 100000

portfolio = []

day = []

        

for idx, val in enumerate(y_pred):

    # if model suggests SELL

    if val == False:

    # increase the bank by the ETH price * number of ETH held

        bank = bank + (ETH * df_test["ETH_price"].iloc[idx])

        # reset the amount of ETH held

        ETH = 0

        portfolio.append(ETH * df_test["ETH_price"].iloc[idx] + bank)

        day.append(idx)

    # if model suggests BUY

    else:

        # buy ETH with all funds 

        ETH = ETH + bank/df_test["ETH_price"].iloc[idx]

        bank = 0

        portfolio.append(ETH * df_test["ETH_price"].iloc[idx] + bank)

        day.append(idx)

        

fig, ax1 = plt.subplots(figsize=(20, 10))

ax1.plot(portfolio,label="Model portfolio")

ax1.set_ylabel('USD ($)',size=16, rotation = 90)



# Tilt x-axis labels

for tick in ax1.get_xticklabels():

    tick.set_rotation(315)

    

# Graph cost-averaging



cost_avg_bank = 100000

cost_avg_ETH = 0

cost_avg_portfolio = []



for i in range(45):

    cost_avg_bank = cost_avg_bank - (100000/45)

    cost_avg_ETH = cost_avg_ETH + (100000/45)/df_test["ETH_price"].iloc[i]

    cost_avg_portfolio.append(cost_avg_bank+cost_avg_ETH*df_test["ETH_price"].iloc[i])

    

plt.plot(cost_avg_portfolio, label="Cost averaging")



# Graph buying all first day



hodl_ETH = 100000/df_test["ETH_price"].iloc[0]

hodl_portfolio = []



for i in range(45):

    hodl_portfolio.append(hodl_ETH*df_test["ETH_price"].iloc[i])

    

plt.plot(hodl_portfolio, label ="All-in day 1")



# Graph random model's suggestions



# generate random arrays

from numpy.random import seed

from numpy.random import randint

# seed random number generator

seed(1)

# generate some integers

random_bools = randint(0, 2, 45) > 0



rand_portfolio = []

rand_ETH = 0

rand_bank = 100000

        

for idx, val in enumerate(random_bools):

    if val == False:

        rand_bank = rand_bank + (rand_ETH * df_test["ETH_price"].iloc[idx])

        rand_ETH = 0

        rand_portfolio.append(rand_ETH * df_test["ETH_price"].iloc[idx] + rand_bank)

    else:

        rand_ETH = rand_ETH + rand_bank/df_test["ETH_price"].iloc[idx]

        rand_bank = 0

        rand_portfolio.append(rand_ETH * df_test["ETH_price"].iloc[idx] + rand_bank)



plt.plot(rand_portfolio,label="Random decisions")

plt.legend(loc="upper left",fontsize=14)

plt.show()
investing_approaches = [portfolio,cost_avg_portfolio,hodl_portfolio,rand_portfolio]

for i in investing_approaches:

    print(i[len(i)-1])
# Bitcoin Cash prices the next day had even stronger correlation with BTC fees than ETH or LTC



df = pd.DataFrame()

df["fees"] = fees

df["BCH_price"] = bch_historical["Close**"].iloc[::-1].reset_index(drop=True)

df["BCH_price_next_day"] = df["BCH_price"].shift(-1)

df.drop([106],axis=0,inplace=True)

df



pearsonr(df["fees"],df["BCH_price_next_day"])
cols = ["fees"]

X = df[cols]

y = df["BCH_price_next_day"].shift(-1) > df["BCH_price"]



model = LogisticRegression()

model.fit(X,y)



X_test = df_test[cols]

y_test = df_test["BCH_price"] - df_test["BCH_price"].shift(-1)

y_test.iloc[-1] = True

y_pred = model.predict(X_test)



y_test = y_test > 0



model.score(X_test,y_test)