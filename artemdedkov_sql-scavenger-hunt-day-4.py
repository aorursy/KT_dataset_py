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
# Question 1: How many Bitcoin transactions were made each day in 2017?

txns_2017 = """with transactions as (select date(timestamp_millis(timestamp)) Date,
                    transaction_id txn_id
                    from `bigquery-public-data.bitcoin_blockchain.transactions`
                    where extract(year from (date(timestamp_millis(timestamp)))) = 2017
                )
                
                select Date,
                count(*) No_of_txns
                from transactions
                group by Date
                order by Date """
txns_data = bitcoin_blockchain.query_to_pandas_safe(txns_2017, max_gb_scanned=3)
# Checking data
txns_data.head()
plt.plot(txns_data.Date, txns_data.No_of_txns)
plt.title('Number of Transactions Each Day in 2017');
# Question 2: How many transactions are associated with each merkle root?
merkle_query = """select merkle_root,
                count(*) No_of_txns
                from `bigquery-public-data.bitcoin_blockchain.transactions`
                group by merkle_root
                order by count(*) desc
                """

merkle_data = bitcoin_blockchain.query_to_pandas_safe(merkle_query, max_gb_scanned = 19)
# Top 10 merkle trees by transactions co
merkle_data.head(10)