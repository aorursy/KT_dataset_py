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
#Q1. How many Bitcoin transactions were made each day in 2017?
query1 = """ WITH time AS 
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
            Having year=2017
            ORDER BY day
        """
#bitcoin_blockchain.estimate_query_size(query1)
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
print(transactions_per_day)

# import plotting library
#import matplotlib.pyplot as plt
#sample_1=plt.plot(transactions_per_day.transactions)

#Q2 How many transactions are associated with each merkle root?¶

query2="""SELECT COUNT(transaction_id) AS transaction,
                 merkle_root AS root
          FROM `bigquery-public-data.bitcoin_blockchain.transactions`
          GROUP BY root
          ORDER BY transaction DESC
"""

bitcoin_blockchain.estimate_query_size(query2)

merkle = bitcoin_blockchain.query_to_pandas(query2)

print(merkle)

sample_b=plt.plot(merkle.transaction)
#sample_b=plt.title('Merkle transaction in Yr')