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
import bq_helper

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="biqquery-public-data",
                                             dataset_name="bitcoin_blockchain")
query_per_day="""WITH time as
            (
                SELECT TIMESTAMP_MILLIS(timestamp) As trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT 
                COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) =2017
            GROUP BY day, year
            ORDER BY day"""

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_per_day,max_gb_scanned=30)
transactions_per_day
# import plotting library
import matplotlib.pyplot as plt
# plot monthly bitcoin transactions
plt.rcParams["figure.figsize"] = (10,6)
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
query_merkle="""WITH merkle as
            (
                SELECT merkle_root,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT 
                COUNT(transaction_id) AS transactions,
                merkle_root
            FROM merkle
            GROUP BY merkle_root
            ORDER BY transactions DESC"""

transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query_merkle,max_gb_scanned=40)
transactions_per_merkle
# import plotting library
import matplotlib.pyplot as plt
# plot monthly bitcoin transactions
plt.rcParams["figure.figsize"] = (10,6)
plt.plot(transactions_per_merkle.transactions)
plt.title("Transcations Per Merkle")