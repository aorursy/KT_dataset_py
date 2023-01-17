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
                EXTRACT(DAYOFYEAR FROM trans_time) AS day
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day
            ORDER BY day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)


transactions_per_day.transactions.plot()
query = """ 
            SELECT COUNT(transaction_id) AS transactions, merkle_root AS root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY root
            ORDER BY transactions DESC
        """
root_transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=50)
root_transactions
root_transactions.head().plot.bar('root', 'transactions')