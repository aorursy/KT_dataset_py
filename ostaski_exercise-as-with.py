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
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day
            ORDER BY transactions
        """

bitcoin_blockchain.estimate_query_size(query)
# 21.496007729321718, so need to bump the max_gb_scanned in query up to 22 gb

# run a "safe" query and store the resultset into a dataframe
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)

# taking a look
print(transactions_per_day_2017)
# day 14 had the most transaction and day 31 had the least

# saving this in case we need it later
transactions_per_day_2017.to_csv("transactions_per_day_2017.csv")

# the query
query = """ SELECT merkle_root, 
                COUNT(transaction_id) AS transactions_count 
            FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
            GROUP BY merkle_root
            ORDER BY transactions_count
        """

# estimate query size
bitcoin_blockchain.estimate_query_size(query)
# 38.34423000365496, so bump the max_gb_scanned in query up to 39 gb

# run a "safe" query and store the resultset into a dataframe
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=39)

# taking a look
print(transactions_per_merkle_root)
# transactions_count ranges from 1 to 12239 over 519554 merkle_roots

# saving this in case we need it later
transactions_per_merkle_root.to_csv("transactions_per_merkle_root.csv")