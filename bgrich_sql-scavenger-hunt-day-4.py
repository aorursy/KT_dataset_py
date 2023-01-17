# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.table_schema("transactions")
bitcoin_blockchain.head("transactions")
query_trans_per_day = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            GROUP BY day
            ORDER BY day
        """
bitcoin_blockchain.estimate_query_size(query_trans_per_day)
trans_per_day = bitcoin_blockchain.query_to_pandas_safe(query_trans_per_day, max_gb_scanned=21)
print(trans_per_day)
query_trans_per_merkle_root = """SELECT COUNT(transaction_id) AS transactions,
                                        merkle_root
                                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                 GROUP BY merkle_root
                                 ORDER BY transactions DESC
                              """
bitcoin_blockchain.estimate_query_size(query_trans_per_merkle_root)
trans_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query_trans_per_merkle_root, max_gb_scanned=38)
print(trans_per_merkle_root.head())