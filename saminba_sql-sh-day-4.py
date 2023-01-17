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
myquery0 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions, EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year
        """

# note that max_gb_scanned is set to 21, rather than 1
tempdata = bitcoin_blockchain.query_to_pandas_safe(myquery0, max_gb_scanned=21)
tempdata.year.unique()
# How many Bitcoin transactions were made each day in 2017?
myquery1 = """ WITH time AS 
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
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY  year, month,day
            ORDER BY year, month, day
        """

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(myquery1, max_gb_scanned=21)
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")

transactions_per_day.head()
#How many transactions are associated with each merkle root?
#You can use the "merkle_root" and "block_id" columns in the "blocks" table to answer this question.  
myquery2="""SELECT COUNT(transaction_id) AS transactions, merkle_root AS merkle
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY merkle
           ORDER BY transactions DESC
        """
transaction_merkle = bitcoin_blockchain.query_to_pandas(myquery2)

transaction_merkle.head()