import bq_helper

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

query1 = """ WITH bitcoin_per_day AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM bitcoin_per_day
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day 
            ORDER BY year, month, day
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
transactions_per_day
import bq_helper

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

query2 = """ WITH blocks_per_merkle AS 
            (
                SELECT block_id, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            )
            SELECT COUNT(block_id) AS num_blocks, merkle_root AS merkle
            FROM blocks_per_merkle
            GROUP BY merkle
            ORDER BY merkle
        """
blocks_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
blocks_merkle