# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# Your Code Here
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
            EXTRACT(month FROM trans_time) AS month,
                EXTRACT(day FROM trans_time) AS day
            FROM time
            GROUP BY month, day
            ORDER BY month, day
        """
res = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)
print(res)
# Your Code Here

query = """  
            SELECT COUNT(transaction_id) AS transactions,
            merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root 
        """
res = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=39)
print(res)
