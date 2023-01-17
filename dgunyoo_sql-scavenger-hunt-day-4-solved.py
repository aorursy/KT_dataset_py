import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                COUNT(transaction_id) AS transactions,
            FROM time
            GROUP BY day
            ORDER BY day
        """
print('Q1\n',bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21))

query = """ SELECT merkle_root, COUNT(transaction_id) AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
        """
print('\nQ2\n',bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40))