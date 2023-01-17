# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head("transactions")
query = """WITH time AS
        (
            SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
        )
        SELECT COUNT(transaction_id) AS transactions, 
            EXTRACT(MONTH FROM transaction_time) AS month,
            EXTRACT(DATE FROM transaction_time) AS date
        FROM time
        GROUP BY month, date
        ORDER BY month, date
        """
bitcoin_blockchain.estimate_query_size(query)
transactions_per_date = bitcoin_blockchain.query_to_pandas_safe(query=query, max_gb_scanned=24)
transactions_per_date.head(10)
import seaborn as sns
import  matplotlib.pyplot as plt
plt.plot(transactions_per_date.transactions)
plt.title("Dayly Bitcoin Transcations in year 2017")
query = """SELECT merkle_root, COUNT(transaction_id) AS transactions 
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions DESC
            """
bitcoin_blockchain.estimate_query_size(query)
transactions_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query=query, max_gb_scanned=42)
transactions_by_merkle_root.head()