import numpy as np

import pandas as pd



# import package with helper functions 

import bq_helper



# create a helper object for this dataset

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.table_schema('transactions')
query = """ WITH time AS 

            (

                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,

                    transaction_id

                FROM `bigquery-public-data.bitcoin_blockchain.transactions`

            )

            SELECT COUNT(transaction_id) AS transactions,

                EXTRACT(MONTH FROM trans_time) AS month,

                EXTRACT(YEAR FROM trans_time) AS year

            FROM time

            GROUP BY year, month 

            ORDER BY year, month

        """



# note that max_gb_scanned is set to 21, rather than 1

transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=24)
# import plotting library

import matplotlib.pyplot as plt



# plot monthly bitcoin transactions

plt.plot(transactions_per_month.transactions)

plt.title("Monthly Bitcoin Transcations")
# How many Bitcoin transactions were made each day in 2017?



query = """ with trans_day as 

(

select transaction_id, TIMESTAMP_MILLIS(timestamp) as trans_time

from `bigquery-public-data.bitcoin_blockchain.transactions`

)



select extract(year from trans_time) as year, 

extract(month from trans_time) as month,

extract(day from trans_time) as day, 

count(transaction_id) as counts



from trans_day

group by day, year, month

having year = 2017

order by day, month"""



transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=24)

transactions_per_day
# import plotting library

import matplotlib.pyplot as plt



# plot monthly bitcoin transactions

from matplotlib.pyplot import figure

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(transactions_per_day)

plt.title("Daily Bitcoin Transcations")
# How many transactions are associated with each merkle root?



query = """ 

select count(transaction_id) as counts, merkle_root

from `bigquery-public-data.bitcoin_blockchain.transactions`

group by merkle_root

order by counts desc"""



merkel_trans = bitcoin_blockchain.query_to_pandas_safe(query,max_gb_scanned = 42)

merkel_trans
plt.plot(merkel_trans.counts)

plt.title('Transactions associated with each Merkle root')