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
queryDays2017 = """ WITH days AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month
            FROM days
            GROUP BY month, day 
            ORDER BY month, day
        """

transactionsPerDay2017 = bitcoin_blockchain.query_to_pandas_safe(queryDays2017, max_gb_scanned=21)

queryMerkle = """WITH Merkle AS
              (
                  SELECT transaction_id,
                         merkle_root
                  FROM `bigquery-public-data.bitcoin_blockchain.transactions`
              )
              SELECT COUNT(transaction_id) AS transactions,
                  merkle_root              
              FROM Merkle
              GROUP BY merkle_root
              """
                  
transactionsPerMerkle = bitcoin_blockchain.query_to_pandas_safe(queryMerkle, max_gb_scanned=37)