# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ WITH time AS 
            (
                SELECT DATE(TIMESTAMP_MILLIS(timestamp)) AS properTime,
                    transaction_id AS transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            
            SELECT properTime AS day, COUNT(transactions) AS dailyTransactions
            FROM time
            WHERE EXTRACT(YEAR FROM properTime) = 2017
            GROUP BY day
            ORDER BY day
        """

# note that max_gb_scanned is set to 21, rather than 1
transPerDay = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

transPerDay.head()
# time to try to plot my first ever graph in Python...
# import plotting library
import matplotlib.pyplot as plt

# plot daily bitcoin transactions
plt.plot(transPerDay.dailyTransactions)
plt.title("Daily Bitcoin Transcations in 2017")
query2 = """SELECT merkle_root AS merkleRoot, COUNT(transaction_id) AS transPerMerk
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkleRoot
            ORDER BY transPerMerk DESC
        """
# fire up the query
merkles = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)

# cross your fingers...
merkles
# basic plot
plt.boxplot(merkles.transPerMerk)
plt.title("Distribution of Transactions Per Merkle Root")
plt.ylabel("Transactions")