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
transactions_per_month.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.year, transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, day 
            ORDER BY year, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day.head()
#print(transactions_per_day)
#transactions_per_day.day.count()
# import plotting library
import matplotlib.pyplot as plt
# plot monthly bitcoin transactions
plt.plot(transactions_per_day.day, transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations for the year 2017")
import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

query = """ WITH MR AS
            (
            SELECT transaction_id, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT merkle_root AS M_R, 
                    COUNT(transaction_id) AS Nbr_trans
            FROM MR
            GROUP BY M_R 
            ORDER BY Nbr_trans DESC
        """
Nbr_Trans_Associated_Merkle_Root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
#print(Nbr_Trans_Associated_Merkle_Root)
Nbr_Trans_Associated_Merkle_Root.head()
#Nbr_Trans_Associated_Merkle_Root.Nbr_trans.count()
#Nbr_Trans_Associated_Merkle_Root.M_R.count()
