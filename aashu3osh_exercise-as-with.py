# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.table_schema("transactions")
bitcoin_blockchain.head("transactions")
query = """WITH time AS
            (
                SELECT timestamp_millis(timestamp) AS transtime, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM transtime) AS day,
                EXTRACT(YEAR FROM transtime) AS year
            FROM time
            GROUP BY day,year
            ORDER BY day,year DESC """

how_many = bitcoin_blockchain.query_to_pandas(query)
how_many.head()
query2 = """WITH trans AS
        ( SELECT merkle_root AS merkle, transaction_id
          FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
            SELECT COUNT(transaction_id), merkle
            FROM trans
            GROUP BY merkle
            ORDER BY COUNT(transaction_id) DESC"""
root = bitcoin_blockchain.query_to_pandas(query2)
root.head()