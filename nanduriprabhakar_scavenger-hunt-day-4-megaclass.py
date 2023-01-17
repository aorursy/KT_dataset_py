# import package with helper functions 
# import bq_helper
import bq_helper
# create a helper object for this dataset
# bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="bitcoin_blockchain")

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project='bigquery-public-data',dataset_name='bitcoin_blockchain')
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
# Your code goes here :)

#bitcoin_blockchain.head('transactions')
query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day_of_year
            FROM time
            GROUP BY day_of_year
            ORDER BY count(transaction_id) DESC
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
query3 = """select count(transaction_id) as transactions,merkle_root
            from `bigquery-public-data.bitcoin_blockchain.transactions`
            group by merkle_root
            order by transactions DESC"""

transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=40)

transactions_per_merkle_root.head()
