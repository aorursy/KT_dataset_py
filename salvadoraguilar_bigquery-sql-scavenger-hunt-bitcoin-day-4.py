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
                EXTRACT(DAY FROM trans_time) AS day
            FROM time WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY month, day 
            ORDER BY month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_2017.head()
import matplotlib.pyplot as plt
plt.plot(transactions_2017['transactions'])
query = """ SELECT COUNT(transaction_id) AS transactions, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=36.9)
transactions_merkle_root.head()
