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
bitcoin_blockchain.head('transactions')

#query to list the transactions per day. the group by and order by should be 
#in yymmdd to get the correct order
query = """ WITH time AS 
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
            WHERE EXTRACT(YEAR FROM trans_time)= 2017
            GROUP BY year, month, day 
            ORDER BY year, month, day
        """

#estimating the size of the query
bitcoin_blockchain.estimate_query_size(query)

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day.head()
#output the result to a csv file
transactions_per_day.to_csv("transactions_perday_in_2017.csv")
#plotting the result in a line plot
import matplotlib.pyplot as plt
plt.title('Bitcoin transactions 2017') 
plt.xlabel('Days')
plt.ylabel('Transactions')
plt.plot(transactions_per_day.transactions)
#listing a few rows in blocks table
bitcoin_blockchain.head('blocks')
#selecting blocks grouped by merkle root
query = """SELECT COUNT(block_id) as block_count, merkle_root as merkle
            FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            GROUP BY merkle
            ORDER BY block_count"""

#estimating the size of the query
bitcoin_blockchain.estimate_query_size(query)
# note that max_gb_scanned is set to 21, rather than 1
block_count = bitcoin_blockchain.query_to_pandas_safe(query)
block_count.head()
# Output result to csv
block_count.to_csv('blocks_for_merkleroot.csv')