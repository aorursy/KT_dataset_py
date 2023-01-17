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
query = """ WITH w_day AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                       transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT COUNT(transaction_id) AS transactions,
                   EXTRACT(DAYOFYEAR FROM trans_time) AS day
              FROM w_day
             GROUP BY day 
             ORDER BY day
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.rcParams["figure.figsize"] = (10,6)
plt.plot(transactions_per_day.transactions)
plt.title("Dayly Bitcoin Transcations")
transactions_per_day
query = """ SELECT  merkle_root,
                    COUNT(transaction_id) AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root 
            ORDER BY transactions DESC
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.rcParams["figure.figsize"] = (10,6)
plt.plot(transactions_per_root.transactions)
plt.title("Dayly Bitcoin Transcations")
transactions_per_root