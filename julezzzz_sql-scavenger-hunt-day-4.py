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
# import big query package 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

# Number of unique BTC transactions per day in 2017 

# It's a large query/ doesn't run given the space, but the below will count the number of 
# transactions by date, and the number of transactions per that day. It could be used to
# understand how a particular day is pacing compared to all data we already have for that day number. 
# Next step would be to filter so that it only counts towards the day if it's before the specified date
# (I.E January 1st only has January 1st to compare to)- a running sum of sorts. 

#trans_per_day = """ 
#    WITH daily as (
#        SELECT DISTINCT TIMESTAMP_MILLIS(timestamp) as tstamp, transaction_id 
#        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
#        WHERE extract(year FROM TIMESTAMP_MILLIS(timestamp)) = 2017
#    )
#SELECT DISTINCT EXTRACT(day FROM tstamp) day, extract(date FROM tstamp) date, 
#count(distinct transaction_id)over(partition by EXTRACT(day FROM tstamp)) transactions_per_day,
#count(distinct transaction_id)over(partition by EXTRACT(date FROM tstamp)) transactions_per_date
#    FROM daily
#order by day, date
#"""


trans_per_day = """ 
    WITH daily as (
        SELECT DISTINCT TIMESTAMP_MILLIS(timestamp) as tstamp, transaction_id 
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        WHERE extract(year FROM TIMESTAMP_MILLIS(timestamp)) = 2017
    )
SELECT DISTINCT extract(date FROM tstamp) date, 
count(distinct transaction_id) transactions_per_date
    FROM daily
group by date
order by date
"""

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(trans_per_day, max_gb_scanned=21)
print(transactions_per_day.head())

# This will count the number of IDs per merkle root.  
trans_per_root = """ 
    WITH merkle as (
        SELECT DISTINCT merkle_root root, transaction_id 
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
SELECT DISTINCT root, 
count(distinct transaction_id) transactions_per_root
    FROM merkle
group by root
order by root
"""

transactions_per_root = bitcoin_blockchain.query_to_pandas_safe(trans_per_root)
print(transactions_per_root.head())


# This will bucket the number of IDs per merkle root, so you can see the distribution. 
avg_trans_per_root = """ 
    WITH merkle as (
        SELECT DISTINCT merkle_root root, count(distinct transaction_id) id 
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        group by root
    )
SELECT DISTINCT id, 
count(distinct root) no_of_roots_per_group_of_ID
    FROM merkle
group by id
order by id
"""

avg_transactions_per_root = bitcoin_blockchain.query_to_pandas_safe(avg_trans_per_root)
print(avg_transactions_per_root.head())