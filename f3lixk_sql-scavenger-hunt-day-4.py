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
import matplotlib.pyplot as plt

q_day = """WITH time AS
           (
               SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, 
                      transaction_id AS id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT COUNT(id) as num, 
                  EXTRACT(DAY FROM trans_time) AS day,
                  EXTRACT(MONTH FROM trans_time) AS month,
                  EXTRACT(YEAR FROM trans_time) AS year
           FROM time
           GROUP BY year, month, day
           HAVING year = 2017
           ORDER BY month, day           
        """
trans_per_day_17 = bitcoin_blockchain.query_to_pandas_safe(q_day, max_gb_scanned=21)
print(trans_per_day_17)

plt.plot(trans_per_day_17.num)
q_merk = """WITH time AS
            (
               SELECT merkle_root AS merk, 
                      transaction_id AS id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(id),merk AS num
            FROM time
            GROUP BY merk
         """
q_merk = bitcoin_blockchain.query_to_pandas_safe(q_merk, max_gb_scanned=37)
print(q_merk.head())