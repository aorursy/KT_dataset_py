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
# Transactions per day
query1= """WITH time AS
            (SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`)
            SELECT COUNT(transaction_id), EXTRACT(DAY from trans_time) AS day
            FROM time
            GROUP BY day"""
trans_per_day=bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned =21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(trans_per_day.f0_)
plt.title("Daily Bitcoin Transcations in 2017")
#Transaction Associations
query2="""SELECT COUNT(transaction_id) AS transactions, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions DESC"""
trans_assoc= bitcoin_blockchain.query_to_pandas_safe(query2,max_gb_scanned=37)
trans_assoc.head()
#Transaction Associations per Day
query3=""" WITH time AS
           (SELECT TIMESTAMP_MILLIS(timestamp) as ttime,transaction_id, merkle_root
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`) 
            SELECT COUNT(transaction_id) AS transactions, merkle_root, EXTRACT(DAY FROM ttime) AS Day
            FROM time
            GROUP BY Day, merkle_root"""
trans_assoc= bitcoin_blockchain.query_to_pandas_safe(query3,max_gb_scanned=40)
trans_assoc.head()