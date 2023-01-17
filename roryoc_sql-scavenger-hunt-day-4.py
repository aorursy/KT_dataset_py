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
# Your code goes here :)
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT
                EXTRACT(DATE FROM trans_time)  AS trans_date,
                COUNT(transaction_id) as trans_no
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time)  = 2017
            GROUP BY trans_date
            ORDER BY trans_date
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_by_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_by_day
# Your code goes here :)
query = """ WITH merkle_root_analysis AS 
            (
                SELECT  merkle_root,
                    count(transaction_id) as trans_count
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
            )
            SELECT
                merkle_root,
                trans_count
            FROM merkle_root_analysis
            ORDER BY merkle_root
        """

# note that max_gb_scanned is set to 21, rather than 1
merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
merkle_root