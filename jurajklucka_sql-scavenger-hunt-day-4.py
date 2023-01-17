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
transactions_per_month.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
query = """ WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            
            )
            SELECT DATE(EXTRACT(YEAR FROM trans_time), 
                        EXTRACT(MONTH FROM trans_time), 
                        EXTRACT(DAY FROM trans_time)) AS date, 
                    COUNT(transaction_id) AS transactions
                
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time)=2017
            GROUP BY date
            ORDER BY date
"""

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day.head()
import matplotlib.pyplot as plt

plt.subplots(figsize=(15,7))
plt.plot(transactions_per_day.date, transactions_per_day.transactions)
plt.title('2017 Bitcoin Transactions')


bitcoin_blockchain.table_schema("transactions")
bitcoin_blockchain.head("transactions", selected_columns="merkle_root",num_rows=10)


query = """SELECT DISTINCT (merkle_root) as merkle_roots,
                COUNT(transaction_id) AS transactions
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY 1
           ORDER BY 2 DESC
        """
transaction_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query,max_gb_scanned=38)
transaction_by_merkle_root.head()
