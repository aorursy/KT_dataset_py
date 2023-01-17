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
# print information on all the columns in the "transactions" table 
# in the bitcoin_blockchain dataset
bitcoin_blockchain.table_schema("transactions")
# preview the first couple of lines of the "transactions" table
bitcoin_blockchain.head("transactions")
# query to find out the number of transactions per day in 2017
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DATE FROM trans_time) AS date
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY date
            ORDER BY date
        """

# check how big this query will be
bitcoin_blockchain.estimate_query_size(query)
# save the results returned into a dataframe 'transactions_per_day_2017'
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas(query)
# plot daily bitcoin transactions
plt.plot(transactions_per_day_2017.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
# look into the first few rows of the dataframe 'transactions_per_day_2017'
transactions_per_day_2017.head()
# look into the last few rows of the dataframe 'transactions_per_day_2017'
transactions_per_day_2017.tail()
# Look into the summary stats of the transactions_per_day dataframe
transactions_per_day_2017.describe()
# save our dataframe as a .csv 
transactions_per_day_2017.to_csv("transactions_per_day_2017.csv")
# preview the first ten entries in the merkle_root column of the transactions table
bitcoin_blockchain.head("transactions", selected_columns="merkle_root", num_rows=10)
# query to find out the number of transactions associated with each merkle root
query = """SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions
        """

# check how big this query will be
bitcoin_blockchain.estimate_query_size(query)
# save the results returned into a dataframe 'transactions_per_merkle_root'
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas(query)
# plot bitcoin transactions per merkle root
plt.plot(transactions_per_merkle_root.transactions)
plt.title("Transcations per Merkle root")
# look into the first few rows of the dataframe 'transactions_per_merkle_root'
transactions_per_merkle_root.head()
# look into the last few rows of the dataframe 'transactions_per_merkle_root'
transactions_per_merkle_root.tail()
# Look into the summary stats of the transactions_per_merkle_root dataframe
transactions_per_merkle_root.describe()