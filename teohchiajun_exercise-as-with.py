# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# build query
query = """WITH time AS
           (
               SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                   transaction_id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT COUNT(transaction_id) AS transactions,
               EXTRACT(DAY FROM trans_time) AS Day,
               EXTRACT(MONTH FROM trans_time) AS Month,
               EXTRACT(YEAR FROM trans_time) AS Year
           FROM time
           GROUP BY Year, Month, Day
           ORDER BY Year, Month, Day
        """
# transform to dataframe
trans_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 22)
trans_per_day.head()
# import pyplot
import matplotlib.pyplot as plt

# plot daily transactions
plt.plot(trans_per_day.transactions)
plt.title('Daily Bitcoin Transaction')
# build query
query_2 = """SELECT merkle_root, 
                    COUNT(transaction_id) AS transaction
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
                ORDER BY transaction DESC
          """
trans_per_root = bitcoin_blockchain.query_to_pandas_safe(query_2, max_gb_scanned = 40)
trans_per_root.head()