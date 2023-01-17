# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# Your Code Here
query = """ WITH day2017 AS
            (
                WITH time AS 
                (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                        transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(transaction_id) AS transactions,
                    EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                    EXTRACT(YEAR FROM trans_time) AS year
                FROM time
                GROUP BY year, day 
                ORDER BY year, day
            )
            SELECT transactions, day
            FROM day2017
            WHERE year = 2017
        """
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=25)
transactions_per_day_2017
# Your Code Here
query = """SELECT COUNT(transaction_id) as num, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY num
        """
table2 = bitcoin_blockchain.query_to_pandas(query)
table2.shape
table2.head(20)
table2.tail(20)