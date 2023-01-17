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
year_query = """ 
            WITH time AS 
                (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS YEAR,
                        transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
                )
            SELECT YEAR,
                EXTRACT(DAYOFYEAR FROM trans_time) AS DAY_OF_THE_YEAR,
                COUNT(transaction_id) AS TX_COUNT
            FROM time
            GROUP BY YEAR, DAY_OF_THE_YEAR
            ORDER BY DAY_OF_THE_YEAR            
        """
bitcoin_blockchain.estimate_query_size(year_query)
df_transactions_2017= bitcoin_blockchain.query_to_pandas_safe(year_query, max_gb_scanned=21)
df_transactions_2017

plt.plot(df_transactions_2017.TX_COUNT)
plt.title("Daily Bitcoin Transcations in 2017")
merkle_query = """    
                SELECT merkle_root,
                       COUNT(transaction_id) as TX_COUNT
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`                    
                GROUP BY merkle_root
                ORDER BY TX_COUNT DESC
              """
df_merkel = bitcoin_blockchain.query_to_pandas_safe(merkle_query, max_gb_scanned=37)
df_merkel
plt.plot(df_merkel.TX_COUNT)
plt.title("Transactions associated with Merkle")