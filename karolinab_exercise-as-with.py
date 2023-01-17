# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head("transactions")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day       
                FROM time
            GROUP BY day
            ORDER BY transactions
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=24)
transactions_per_day
query = """SELECT COUNT(transaction_id) AS tran, merkle_root
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        ORDER BY tran
         """
merkle = bitcoin_blockchain.query_to_pandas(query)
merkle
