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
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
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
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT 
            DATE(trans_time) as DATE_ID,
            COUNT(transaction_id) AS transactions
            FROM time
            GROUP BY DATE_ID 
            ORDER BY DATE_ID
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# cost of query
print("Query size estimate:")
print(bitcoin_blockchain.estimate_query_size(query))
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head("transactions")
transactions_per_day_2017
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day_2017.DATE_ID, transactions_per_day_2017.transactions)
plt.title("Daily Bitcoin Transactions in 2017")
bitcoin_blockchain.head("blocks")
query2b = """ WITH mini_blocks AS 
            (
                SELECT merkle_root, transaction_id as tran_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
            SELECT 
                merkle_root, 
                count(tran_id) as tran_id_cnt
            FROM mini_blocks
            GROUP BY merkle_root 
            ORDER BY tran_id_cnt desc
        """

# note that max_gb_scanned is set to 21, rather than 1
tran_id_cnt_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2b, max_gb_scanned=37)
tran_id_cnt_per_merkle.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(tran_id_cnt_per_merkle.tran_id_cnt)
plt.title("Transactions count per merkle ")
# fast query
# how to measure the time it takes to run?
query1a = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT 
            DATE(trans_time) as DATE_ID,
            COUNT(transaction_id) AS transactions
            FROM time
            GROUP BY DATE_ID 
            ORDER BY DATE_ID
        """
# cost of query
print("Query size estimate:")
print(bitcoin_blockchain.estimate_query_size(query1a))

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day_2017_1a = bitcoin_blockchain.query_to_pandas_safe(query1a, max_gb_scanned=21)
# slow query
# run time?
query1b = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
              
            )
            SELECT 
            DATE(trans_time) as DATE_ID,
            COUNT(transaction_id) AS transactions
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY DATE_ID 
            ORDER BY DATE_ID
        """
# cost of query
print("Query size estimate:")
print(bitcoin_blockchain.estimate_query_size(query1b))

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day_2017_1b = bitcoin_blockchain.query_to_pandas_safe(query1b, max_gb_scanned=21)