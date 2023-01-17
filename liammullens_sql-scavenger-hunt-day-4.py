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
# Get number of bitcoin transactions per day
query = """
        WITH time AS 
        (
            SELECT TIMESTAMP_MILLIS(timestamp) as trans_time,
            transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id) as Transactions,
        EXTRACT(MONTH FROM trans_time) AS Month,
        EXTRACT(DAY FROM trans_time) AS Day
        FROM time
        GROUP BY Month, Day
        ORDER BY Month, Day
        """
daily_transactions = bitcoin_blockchain.query_to_pandas(query)
print(daily_transactions)
# Get number of bitcoin transactions per merkle root
query = """
        SELECT COUNT(transaction_id) AS Transactions, merkle_root
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        """
trans_by_root = bitcoin_blockchain.query_to_pandas(query)
print(trans_by_root)