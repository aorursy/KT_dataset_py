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
bitcoin_blockchain.list_tables()

bitcoin_blockchain.table_schema("transactions")
bitcoin_blockchain.head("transactions",2)
query_2="""

with btc as
(
select  TIMESTAMP_MILLIS(timestamp) as Time, transaction_id as Transactions
from `bigquery-public-data.bitcoin_blockchain.transactions`
)
select extract(day from Time) as Day_of_trans, count(Transactions) as Trans
from btc
group by Day_of_trans
order by Day_of_trans asc

        """
bitcoin_blockchain.estimate_query_size(query_2)
btc_query2=bitcoin_blockchain.query_to_pandas_safe(query_2,max_gb_scanned=21)
print(btc_query2)
btc_query2.to_csv("btc_query2")
query_3="""
            select count( transaction_id) as Transactions, merkle_root
            from `bigquery-public-data.bitcoin_blockchain.transactions`
            group by merkle_root

        """
bitcoin_blockchain.estimate_query_size(query_3)
btc_query3=bitcoin_blockchain.query_to_pandas(query_3)
print(btc_query3)
btc_query3.to_csv("btc_query3")