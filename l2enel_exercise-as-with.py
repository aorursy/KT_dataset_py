# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head('transactions', 3)
query = """WITH transactions_2017 AS
            ( SELECT transaction_id, TIMESTAMP_MILLIS(timestamp) AS time
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id), EXTRACT(day from time) AS day
            FROM transactions_2017
            WHERE EXTRACT(year from time) = 2017
            GROUP BY day
            ORDER BY day"""

bitcoin_blockchain.estimate_query_size(query)
bitcoin_blockchain.query_to_pandas(query)
query2 = """ WITH dates AS (
    SELECT TIMESTAMP_MILLIS(timestamp) as time
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
    SELECT time
    FROM dates
    WHERE EXTRACT(year from time) = 2017
    """

#bitcoin_blockchain.estimate_query_size(query2)
#bitcoin_blockchain.query_to_pandas(query2)