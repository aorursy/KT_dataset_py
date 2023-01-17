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
bitcoin_blockchain.list_tables()
bitcoin_blockchain.table_schema('transactions')
bitcoin_blockchain.head('transactions')
query = """WITH time AS
           (
               SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time,
               transaction_id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT EXTRACT(DAYOFYEAR FROM transaction_time) AS day,
           COUNT(transaction_id) as num_transactions
           FROM time
           WHERE EXTRACT(YEAR FROM transaction_time) = 2017
           GROUP BY day
           ORDER BY day
           
        """
bitcoin_blockchain.estimate_query_size(query)
transactions_in_2017 = bitcoin_blockchain.query_to_pandas(query)
plt.plot(transactions_in_2017['num_transactions'])
plt.xlabel('Day of 2017')
plt.ylabel('# of Bitcoin Transactions')
# 2. HOw many transactions are associated with each merkle root
query = """
           SELECT merkle_root, COUNT(transaction_id) AS num_transactions
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY merkle_root
        """
bitcoin_blockchain.estimate_query_size(query)
root_data = bitcoin_blockchain.query_to_pandas(query)
root_data.tail()
plt.plot(root_data['num_transactions'])
plt.yscale('log')
plt.plot(sorted(root_data['num_transactions']))
plt.yscale('log')
plt.ylabel('# Transactions in a Merkle Root')