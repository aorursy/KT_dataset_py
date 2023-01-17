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
transactions_per_month
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)
bitcoin_blockchain.head("transactions")


query_1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(YEAR FROM trans_time) AS year,
                EXTRACT(MONTH FROM trans_time) AS month
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time)=2017
            GROUP BY year,month,day
            ORDER BY year, month,day
            
        """
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas(query_1)
transactions_per_day_2017

plt.plot(transactions_per_day_2017.transactions)
plt.title("Transactions per day for the year of 2017")
plt.show()
query_2 = """ WITH merkle AS 
            (
                SELECT transaction_id,merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,merkle_root as merkle
                
            FROM merkle
            GROUP BY merkle
            ORDER BY transactions
            
        """
transactions_per_merkle = bitcoin_blockchain.query_to_pandas(query_2)
transactions_per_merkle
#plt.plot(transactions_per_merkle.transactions,transactions_per_merkle.merkle)
#plt.show()

