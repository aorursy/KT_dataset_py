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
                EXTRACT(DAYOFWEEK FROM trans_time) AS day
            FROM time
            GROUP BY day 
            ORDER BY transactions DESC
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=0.1)
bitcoin_blockchain.table_schema("transactions")
bitcoin_blockchain.head("transactions")
query = """ WITH merkle AS
            (   SELECT transaction_id,merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                   merkle_root
            FROM merkle
            GROUP BY merkle_root 
            ORDER BY transactions DESC
        """
transactions_bymerkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=0.1)