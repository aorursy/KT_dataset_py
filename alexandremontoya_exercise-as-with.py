# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables() 
bitcoin_blockchain.head("transactions")
query = """
            WITH trans_time AS
            (
            SELECT transaction_id AS id, 
                   TIMESTAMP_MILLIS(timestamp) AS time
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )

            SELECT EXTRACT(YEAR FROM time) AS year, 
                   COUNT(id) AS num_trans
            FROM trans_time
            WHERE EXTRACT(YEAR FROM time) = 2017
            GROUP BY EXTRACT(YEAR FROM time)
        """

bitcoin_transactions_per_year = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)
print(bitcoin_transactions_per_year)
query = """
        SELECT merkle_root, COUNT(transaction_id) as total_trans
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        ORDER BY total_trans DESC
        """
transactions_with_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=39)
print(transactions_with_merkle.head())