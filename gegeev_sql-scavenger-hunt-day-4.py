# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query_dayly_tr = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                EXTRACT(YEAR FROM trans_time) AS year
                
            FROM time
            GROUP BY year,day 
            HAVING year=2017
            ORDER BY year,day
         """
#
dayly_transaction=bitcoin_blockchain.query_to_pandas_safe(query_dayly_tr, max_gb_scanned=21)
import matplotlib.pyplot as plt
plt.plot(dayly_transaction.transactions)
plt.title("Daly Bitcoin Transcations")
query_dayly_tr = """ WITH time AS 
            (
                SELECT merkle_root AS mr,
                transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,mr
                          
            FROM time
            GROUP BY mr
        """
bitcoin_blockchain.query_to_pandas_safe(query_dayly_tr, max_gb_scanned=37)