# import package with helper functions 
import bq_helper
import pandas as pd

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
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
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(YEAR FROM trans_time) AS year,
                EXTRACT(MONTH FROM trans_time) AS month,                
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            GROUP BY year, month, day 
            ORDER BY year, month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

#look at the first few rows:
transactions_per_day.head(10)
transactions_per_day['date'] = pd.to_datetime(transactions_per_day.loc[:,['year','month','day']])
transactions_per_day.set_index('date', inplace=True)
transactions_per_day.head(5)
#transactions_per_day.loc[:,['date','transactions']].plot()
plt.figure(figsize=(18,12))
transactions_per_day['transactions'].plot()
query = """ SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root 
            HAVING transactions > 100
            ORDER BY transactions DESC
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
print('Top 25 merkle roots by transactions:')
transactions_per_root.head(25)