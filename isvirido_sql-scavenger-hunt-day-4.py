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
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-publi-data",
                                             dataset_name = "bitcoin_blockchain")
query = """WITH dates AS
        (
            SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id,
                     EXTRACT(DAY FROM (TIMESTAMP_MILLIS(timestamp))) as day_numbr,
                     EXTRACT(MONTH FROM (TIMESTAMP_MILLIS(timestamp))) as month_numbr,
                     EXTRACT(YEAR FROM (TIMESTAMP_MILLIS(timestamp))) as year_numbr

                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                
        )
        SELECT year_numbr, month_numbr, day_numbr, COUNT(transaction_id) as trans
        FROM dates
        WHERE year_numbr = 2017
        GROUP BY year_numbr, month_numbr, day_numbr   
        ORDER BY year_numbr, month_numbr, day_numbr
        """
#trans = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
#print(trans)
query2 = """SELECT merkle_root,
                COUNT(transaction_id) as trans
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            WHERE EXTRACT(YEAR FROM (TIMESTAMP_MILLIS(timestamp))) = 2018              
            GROUP BY merkle_root  
       """
trans_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
print(trans_per_merkle)