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
# bitcoin_blockchain.estimate_query_size(query)
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_month.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
bitcoin_blockchain.head("transactions")
query_1 = """WITH tr_2017 AS 
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                        transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
             SELECT COUNT(transaction_id) AS transactions,
                    EXTRACT(MONTH FROM trans_time) AS month,
                    EXTRACT(DAY FROM trans_time) AS day
             FROM tr_2017
             WHERE EXTRACT(YEAR FROM trans_time) = 2017
             GROUP BY month, day
             ORDER BY month, day
             """
bitcoin_blockchain.estimate_query_size(query_1)
transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(query_1, max_gb_scanned=21)
transactions_2017.head()
query_2 = """
          SELECT COUNT(transaction_id) AS transactions,
                 merkle_root
          FROM `bigquery-public-data.bitcoin_blockchain.transactions`
          GROUP BY merkle_root
          ORDER BY 1 DESC
          """
bitcoin_blockchain.estimate_query_size(query_2)
transactions_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query_2, max_gb_scanned=37)
transactions_by_merkle_root.head()