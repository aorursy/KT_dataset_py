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
transactions_per_month2 = bitcoin_blockchain.query_to_pandas(query)
# import plotting library
import matplotlib.pyplot as plt

print(transactions_per_month2)
# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")


query2 = """WITH BTC_trans AS
            (
                SELECT transaction_id,
                    TIMESTAMP_MILLIS(timestamp) as TransTime
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )

            SELECT COUNT(transaction_id) as count_trans,
                EXTRACT(DAYOFWEEK FROM TransTime) as DayOfWeek,
                EXTRACT(YEAR FROM TransTime) as YEAR
            FROM BTC_trans 
              -- WHERE COUNT(transaction_id) is null
            GROUP BY YEAR,DayOfWeek 
            HAVING YEAR = 2017
            ORDER BY count_trans DESC
        """


btc_trans_2017 = bitcoin_blockchain.query_to_pandas(query2)
bitcoin_blockchain.estimate_query_size(query_2)
print(btc_trans_2017)

query3 = """
            SELECT merkle_root,
                COUNT(transaction_id) as CountOfTrans
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
        """

# Estimate query size
bitcoin_blockchain.estimate_query_size(query3)

merkle_root_trans = bitcoin_blockchain.query_to_pandas(query3)
print(merkle_root_trans)