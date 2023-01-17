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
# Your code goes here :)# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

query = """ WITH Tran AS 
            ( 
            SELECT  TIMESTAMP_MILLIS(timestamp) AS Time,
                    transaction_id AS TranID
            FROM    `bigquery-public-data.bitcoin_blockchain.transactions` 
            )
            SELECT  EXTRACT(YEAR FROM Time) AS yr,
                    EXTRACT(MONTH FROM Time) AS mn,
                    EXTRACT(DAY FROM Time) AS dy,
                    COUNT(TranID) AS Tran_Cnt
            FROM Tran
            GROUP BY yr, mn, dy
            HAVING yr = 2017
            ORDER BY yr, mn, dy
        """

tran_cnt = bitcoin_blockchain.query_to_pandas(query)

print(tran_cnt)

