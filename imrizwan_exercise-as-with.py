# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head("blocks")
bitcoin_blockchain.head("transactions")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

transactions_per_year = bitcoin_blockchain.query_to_pandas(query)
print(transactions_per_year)
query_1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT EXTRACT(MONTH FROM trans_time) AS month,
                   EXTRACT(YEAR FROM trans_time) AS year,
                   EXTRACT(DAY FROM trans_time) AS day,
                   COUNT(transaction_id) AS transactions
            FROM time
            GROUP BY year, month, day 
            HAVING year = 2017
            ORDER BY year, month, day
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas(query_1)
print(transactions_per_day)

query_2 = """ SELECT COUNT(transaction_id) as transactions, merkle_root as merkle
              FROM `bigquery-public-data.bitcoin_blockchain.transactions`
              GROUP BY merkle
              ORDER BY transactions 
         """
transactions_merkle_root = bitcoin_blockchain.query_to_pandas(query_2)
print(transactions_merkle_root)
