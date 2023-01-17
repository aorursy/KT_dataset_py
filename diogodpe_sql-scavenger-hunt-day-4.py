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
# How many Bitcoin transactions were made each day in 2017?
query = """ WITH timestamp_converted AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,                
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions_count,
                EXTRACT(DAY FROM trans_time) AS day
            FROM timestamp_converted
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day
            ORDER BY COUNT(transaction_id)
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_month.transactions.head()
#bitcoin_blockchain.estimate_query_size(query)
# How many transactions are associated with each merkle root?
query = """ WITH temp AS 
            (
                SELECT merkle_root,                
                     COUNT(transaction_id) AS transactions_count
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
            )
            SELECT transactions_count,
                merkle_root
            FROM temp
            ORDER BY  transactions_count DESC
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
transactions.count()
#bitcoin_blockchain.estimate_query_size(query)