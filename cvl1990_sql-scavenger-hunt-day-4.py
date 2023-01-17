# import package with helper functions 
import bq_helper

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

## Scavenger hunt

# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

# How many Bitcoin transactions were made each day in 2017?
bitcoin_blockchain.head('transactions')
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
 
           SELECT * FROM(
           SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS DAY,
                EXTRACT(MONTH FROM trans_time) AS MONTH,
                EXTRACT(YEAR FROM trans_time) AS YEAR
            FROM time
            GROUP BY YEAR,MONTH,DAY 
            ORDER BY YEAR,MONTH,DAY
            )AS A1
            WHERE YEAR=2017
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day2017
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day2017.transactions)
plt.title("Dayly Bitcoin Transcations 2017")

# How many transactions are associated with each merkle root?
query2 = """ 
                SELECT 
                merkle_root,
                COUNT(transaction_id) AS trans
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
        """
transaction_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=50)
transaction_merkle

