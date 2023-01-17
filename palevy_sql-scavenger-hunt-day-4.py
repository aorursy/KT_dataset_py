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

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.table_schema("transactions")

bitcoin_blockchain.head("transactions")
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.table_schema("transactions")

query = """ With time as
            (
                Select timestamp_millis(timestamp)as trans_date, 
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            Select Count(transaction_id) AS transactions,
                Extract(day from trans_date)as day,
                Extract(Month from trans_date) as month,
                Extract(Year from trans_date)as year
            FROm time
            where Extract(Year from trans_date) = 2017
            Group by year, month, day
            Order by year, month, day
            """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

bitcoin_blockchain.head("blocks")
#How many blocks are associated with each merkle root?
  #  * You can use the "merkle_root" and "block_id" columns in the "blocks"
    
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

query2= """ Select (count(block_id)/count(merkle_root))
            From `bigquery-public-data.bitcoin_blockchain.blocks`
           
        """
#bitcoin_blockchain.estimate_query_size(query2)
blocks_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)

print(blocks_per_merkle_root)