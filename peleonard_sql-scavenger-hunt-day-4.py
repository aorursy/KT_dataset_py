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
# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head("transactions")
#queryW = """SELECT *
#            FROM `bigquery-public-data.bitcoin_blockchain.transactions`"""
#bitcoinW = bitcoin_blockchain.query_to_pandas_safe(queryW, max_gb_scanned=530)
#bitcoinW[0:10]
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY month, day 
            ORDER BY month, day
        """
# note that max_gb_scanned is set to 21, rather than 1
bitcoin_perday = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
bitcoin_perday[0:10]
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(bitcoin_perday.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
# check the information requested
bitcoin_blockchain.table_schema("transactions")
bitcoin_blockchain.head("transactions")
query2 = """SELECT COUNT(transaction_id) AS transactions, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions DESC"""
trans_perMR = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=38)
trans_perMR[0:10]
