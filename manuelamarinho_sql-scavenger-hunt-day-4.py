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
query7 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM (TIMESTAMP_MILLIS(timestamp))) = 2017

            )
            SELECT DATE(EXTRACT(YEAR FROM transaction_time), 
               EXTRACT(MONTH FROM transaction_time), 
               EXTRACT(DAY FROM transaction_time)) AS date, 
               COUNT(transaction_id) AS transactions
            FROM time
            GROUP BY 1
            ORDER BY 1
            """
bitcoin_blockchain.estimate_query_size(query7)


transactions_per_day = bitcoin_blockchain.query_to_pandas(query7)

transactions_per_day
import matplotlib.pyplot as plt

ax = plt.subplots(figsize=(15,7))
plt.plot(transactions_per_day.date, transactions_per_day.transactions)
plt.title('2017 Bitcoin Transactions')
query8 = """ SELECT COUNT(transaction_id) AS transactions, merkle_root
             FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             GROUP BY 2
             ORDER BY 1 DESC
         """

result2 = bitcoin_blockchain.query_to_pandas(query8)

result2
import matplotlib.pyplot as plt

ax = plt.subplots(figsize=(15,7))
plt.plot(result2.transactions)
plt.title('Transactions')