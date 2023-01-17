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
            SELECT count(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH from trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day 
            ORDER BY year, month, day
            
        """

# note that max_gb_scanned is set to 21, rather than 1
# connection keeps timing out - not the best infrastructure for this kind of task!!!
bitcoin_blockchain.estimate_query_size(query)
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day_2017.head(10)
# import plotting library
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# plot monthly bitcoin transactions
plt.plot(transactions_per_day_2017.transactions)
plt.title("daily Bitcoin Transcations")
query2 = """ WITH mroot AS 
            (
                SELECT merkle_root as m_root, transaction_id AS trans_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT count(trans_id) AS count, m_root
            FROM mroot
            GROUP by m_root
        """

# note that max_gb_scanned is set to 21, rather than 1
# connection keeps timing out - not the best infrastructure for this kind of task!!!
merkle_count = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
merkle_count.head(10)