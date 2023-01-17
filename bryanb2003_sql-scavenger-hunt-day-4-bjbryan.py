# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="bitcoin_blockchain")

bitcoin_blockchain.list_tables()
bitcoin_blockchain.head("transactions")
bitcoin_blockchain.table_schema('transactions')


query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT EXTRACT(DATE FROM trans_time) AS date,
                COUNT(transaction_id) AS transactions
            FROM time
            GROUP BY date 
            ORDER BY date
        """

bitcoin_blockchain.estimate_query_size(query)
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_month
query = """ 
            SELECT merkle_root,
                count(distinct transaction_id) as trans_cnt
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
        """



bitcoin_blockchain.estimate_query_size(query)
merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
merkle