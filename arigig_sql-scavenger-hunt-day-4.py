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
bc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# list all the tables in this dataset "Bitcoin Blockchain"
bc.list_tables()

# print out table structure one by one
bc.table_schema("blocks")
bc.table_schema("transactions")

# print out few records to understand the data
bc.head("transactions")

# How many Bitcoin transactions were made each day in 2017?
# prepare query to bring 2017 day-wise transactions table
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                       transaction_id
                  FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS trn_cnt,
                   EXTRACT(MONTH FROM trans_time) AS month,
                   EXTRACT(DAY FROM trans_time) AS day
              FROM time
             WHERE EXTRACT(YEAR FROM trans_time) = 2017
             GROUP BY month, day
             ORDER BY month, day
        """
# estimate query size before execution
bc.estimate_query_size(query)

# 20.63 GB of space required to run the query
# lets run in safe mode to avoid more space utilization with 21 GB as max size
# we want the result to be stored in a dataframe

daywise_bitcoin_transactions = bc.query_to_pandas_safe(query,max_gb_scanned = 21)

# print out few records out of the dataframe object daywise_bitcoin_transactions
# using head() and tail()
daywise_bitcoin_transactions.head(5)
daywise_bitcoin_transactions.tail(5)

# import plotting library
import matplotlib.pyplot as plt

# plot daily bitcoin transactions
plt.plot(daywise_bitcoin_transactions.trn_cnt)
plt.title("Daily Bitcoin Transcations")

# How many transactions are associated with each merkle root?
# print out the schema structure once again for transactions table
bc.table_schema("transactions")

# print our few records to understand data
bc.head("transactions")

# query to get all the transactions associated with a merkle_root
query = """ SELECT merkle_root, COUNT(transaction_id) AS trn_cnt
              FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             GROUP BY merkle_root
             ORDER BY trn_cnt DESC
        """

# estimate query size before run
bc.estimate_query_size(query)

# 36.82 GB of space required to run the query
# running it in safe mode to avoid space over flow with 37
# store result in a dataframe
merkle_wise_trans = bc.query_to_pandas_safe(query,max_gb_scanned = 37)

# total 509224 merkle roots are available in the table
# with associated transactions varies from 1 to 12239
# print out the dataframe using head and tail
merkle_wise_trans.head(5)
merkle_wise_trans.tail(5)