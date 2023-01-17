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
import bq_helper
t = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                             dataset_name="bitcoin_blockchain")

transactions_day_query = """
with time as(
  select
    timestamp_millis(timestamp) trans_time
  , transaction_id
  from
    `bigquery-public-data.bitcoin_blockchain.transactions`
)
select
  cast(format_timestamp("%F", trans_time) as date) date
, count(transaction_id) transactions
from
  time
group by
  1
order by
  1 asc
"""

transactions_day = t.query_to_pandas_safe(transactions_day_query, max_gb_scanned=21)
print(transactions_day.head(5))
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import to_datetime

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(transactions_day[(transactions_day.date >= '2017-01-01')&(transactions_day.date <= '2017-12-31')].date,
        transactions_day[(transactions_day.date >= '2017-01-01')&(transactions_day.date <= '2017-12-31')].transactions)
ax.set_ylabel('Transactions')
ax.set_xlabel('Date')
ax.set_title('Bitcoin Transactions by Day')
merkle_query = """
with m as (
  select
    merkle_root
  , transaction_id
  from
    `bigquery-public-data.bitcoin_blockchain.transactions`
)
select
  merkle_root
, count(transaction_id) transactions
from
  m
group by
  1
order by
  count(transaction_id) desc
"""

merkle_transactions = t.query_to_pandas(merkle_query)
print(merkle_transactions.head(5))