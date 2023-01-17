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
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month, day
            HAVING year = 2017
            ORDER BY year, month, day
            
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas(query1)
transactions_per_day



import matplotlib.pyplot as plt
plt.bar(range(len(transactions_per_day['transactions'])),transactions_per_day['transactions'])
plt.xlabel("Days of the year 2017")
plt.ylabel("Number of transaction per day")
#plt.scatter(range(len(transactions_per_day['transactions'])),transactions_per_day['transactions'])
# Your code goes here :)
query2 = """
                SELECT COUNT(block_id) AS blocks_per_merkle, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.blocks`
                GROUP BY merkle_root
                ORDER BY blocks_per_merkle DESC
        """

# note that max_gb_scanned is set to 21, rather than 1
blocks_per_merkle = bitcoin_blockchain.query_to_pandas(query2)
blocks_per_merkle