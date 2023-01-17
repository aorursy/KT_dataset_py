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


# how many Bitcoin transaction were made each day in 2017?
queryQ1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                   EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS year,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT COUNT(transaction_id) AS transactions_2017,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day
            FROM time
            GROUP BY day
            ORDER BY day
        """

transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(queryQ1, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day_2017.transactions_2017)
plt.title("Daily Bitcoin Transcations in 2017")
#how many transactions are associated with each Merkle Root?
queryQ2 = """ SELECT COUNT(transaction_id) AS transactions,
                merkle_root AS Merkle
            FROM`bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY Merkle
            ORDER BY transactions  DESC
        """
transactions_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe(queryQ2, max_gb_scanned=37)
# import plotting library
import matplotlib.pyplot as plt

# plot transactions of merkle roots
plt.plot(transactions_by_merkle_root.transactions)
plt.title("Number of Transactions per Merkle Root")
# what are the top transacted merkle roots?
transactions_by_merkle_root.head()
