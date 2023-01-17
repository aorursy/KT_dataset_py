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
#How many Bitcoin transactions were made each day in 2017?
#    * You can use the "timestamp" column from the "transactions" table to answer this question. You can check the [notebook from Day 3](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3/) for more information on timestamps.

query = '''
WITH time AS
(
    SELECT DATE(TIMESTAMP_MILLIS(timestamp)) AS trans_date,
        transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT trans_date AS date, count(*) as n_transactions
FROM time
WHERE EXTRACT(YEAR FROM trans_date) = 2017
GROUP BY date
ORDER BY date
'''
# Interesting note: using date instead of trans_date in the WHERE clause results in an 
# error ("Unrecognized name"), but grouping and ordering by date is fine. Hmm...
per_diem = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
print("Got {} rows in result".format(len(per_diem)))
per_diem.head()
# (Q2 Version 2 - per forums https://www.kaggle.com/questions-and-answers/49820)
# How many *transactions* are associated with each merkle root?

# This was my first attempt at a solution (not using WITH). I thought it
# was quite clever, but unlike the query below, it seems to always result
# in a TimeoutError when I try to run it. I'm not sure why. (Removing the
# ORDER BY doesn't seem to help either)

query = '''
SELECT merkle_root, count(DISTINCT transaction_id) as transactions
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY transactions DESC
'''

if 0:
    # This one needs to scan a little more data still
    per_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=38)
    print("Got {} rows in result".format(len(per_root)))
    per_root.head()
# Another approach to the above (using WITH)
# This one finishes in time (though it's still kind of slow)
query = '''
WITH pairs AS
(
    SELECT merkle_root, transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle_root, transaction_id
)
SELECT merkle_root, count(*) as transactions
FROM pairs
GROUP BY merkle_root
ORDER BY transactions DESC
'''
# This one needs to scan a little more data still
per_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=38)
print("Got {} rows in result".format(len(per_root)))
per_root.head()