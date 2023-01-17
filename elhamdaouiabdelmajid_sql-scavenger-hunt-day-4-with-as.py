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
                EXTRACT(DAYOFYEAR FROM trans_time) AS dayOfYear,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY dayOfYear, year
            HAVING year = 2017
            ORDER BY transactions
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day_2017.head(7)
query = """ WITH trans AS
            (
                SELECT merkle_root,
                       count(transaction_id) AS transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
                ORDER BY transactions DESC
            )
            SELECT *
            FROM trans
            
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
transactions.head(7)