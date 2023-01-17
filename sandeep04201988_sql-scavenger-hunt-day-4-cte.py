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
#plt.plot(transactions_per_month.transactions)
#plt.title("Monthly Bitcoin Transcations")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(Day FROM trans_time) AS Day,
                EXTRACT(Month FROM trans_time) AS Month
            FROM time
            where EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY Day,Month
            ORDER BY Month,Day
        """
bitcoin_blockchain.estimate_query_size(query)
transaction_per_day = bitcoin_blockchain.query_to_pandas_safe(query,max_gb_scanned=21)
transaction_per_day.head()
plt.rcParams["figure.figsize"] = (16,6)
plt.plot(transaction_per_day.transactions)
plt.title('Transactions per day in 2017')
plt.xlabel('Days')
plt.ylabel('Number of Transactions')
plt.show()
bitcoin_blockchain.head('blocks')
query = """with number_blocks as
            (SELECT merkle_root,
                COUNT(DISTINCT block_id) AS blocks
                FROM `bigquery-public-data.bitcoin_blockchain.blocks`
                GROUP BY merkle_root)
            SELECT merkle_root,blocks from number_blocks 
            order by blocks desc            
        """
q2 = bitcoin_blockchain.query_to_pandas_safe(query)
q2