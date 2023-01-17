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
# Question 1 : How many Bitcoin transactions were made each day in 2017?
# The answer is as follows :

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

query1 = """ WITH time1 AS 
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time_1,
                     transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                 WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
             )
            SELECT EXTRACT(YEAR FROM trans_time_1) AS Year,
                EXTRACT(MONTH FROM trans_time_1) AS Month,
                EXTRACT(DAY FROM trans_time_1) AS Day,
                COUNT(transaction_id) AS transactions_every_day
            FROM time1
            GROUP BY Year, Month, Day
            ORDER BY Year, Month, Day
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_every_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
print(transactions_every_day_2017)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_every_day_2017.transactions_every_day)
plt.title("Year_2017 Every Day's Bitcoin Transcations")
plt.xlabel("Every_Day_In_2017")
plt.ylabel("Bitcoin_Transaction_Volume")
# Question 2 : How many transactions are associated with each merkle root?
# The answer is as follows :

query2 = """ WITH merkles AS 
            (
                SELECT merkle_root, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS TransactionQuantity,
                merkle_root AS MerkleRootNumber
            FROM merkles
            GROUP BY merkle_root
            ORDER BY merkle_root
        """
transactions_every_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=38)
print(transactions_every_merkle_root)
transactions_every_day_2017.to_csv("transactions_every_day_2017_UseBitcoinBlockchainDataset.csv")
transactions_every_merkle_root.to_csv("transactions_every_merkle_root_UseBitcoinBlockchainDataset.csv")