# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# query to find out the number of transactions per day in 2017
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            ),
            year AS
            (
                SELECT trans_time, transaction_id 
                FROM time
                WHERE EXTRACT(YEAR FROM trans_time) = 2017
            )
            SELECT EXTRACT(DAYOFYEAR FROM trans_time) as day_of_year,
            COUNT(transaction_id) as num_trans
            FROM year
            GROUP BY EXTRACT(DAYOFYEAR FROM trans_time)
        """

transactions_by_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=24)
transactions_by_day
# query to find out the number of transactions per merkle root
query = """ SELECT merkle_root,
            COUNT(transaction_id) as num_trans
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY merkle_root
        """

transactions_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=42)
transactions_by_merkle_root