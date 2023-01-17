# Your code goes here :)
# How many Bitcoin transactions were made each day in 2017?
# NB: after doing it for 2017, I wanted to compare with 2016, and then with 2015
# to look at the incredible increase of transactions as reported in the press recently.
# Well, that resulted in a LOT of code duplication, so I decided it was time to practice some
# basic Python and mathlib. Here's the result.

import bq_helper
import matplotlib.pyplot as plt

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

years_of_interest = ['2017', '2016', '2015']
plt.title("Daily Bitcom Transcations")

for year in years_of_interest:
  query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day_of_year
            FROM time
            where extract(YEAR FROM trans_time) = """ + year + """ 
            GROUP BY day_of_year
            ORDER BY day_of_year
        """
  # Check the size of the query first and then set max_gb_scanned appropriately
  # Value returned is in GB
  size = bitcoin_blockchain.estimate_query_size(query)
  transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=size + 1)

  # plot daily bitcoin transactions for current year
  plt.plot(transactions_per_day.transactions, label=year)
plt.legend()  
# How many transactions are associated with each merkle root?

import bq_helper
import matplotlib.pyplot as plt

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ SELECT COUNT(transaction_id) AS transactions, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
        """

# Check the size of the query first and then set max_gb_scanned appropriately
size = bitcoin_blockchain.estimate_query_size(query)
transactions_per_mr = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=size + 1)

# plot bitcoin transactions per merkle_root
# NB: There are a little over 500,000 unique Merkle roots in the table. The values used
# for the x-axis are not the actual Merkle root values, but rather, a running ID for
# the 500,000+ unique values. That's a good thing, but it's kind of magic. How does matplotlib
# 'know' to do this? Now having looked at the DB schema, I see that the merkle_root column is 
# string-valued - i.e. those are not hex representations of integers, but just the equivalent strings.
# Maybe matplotlib, requiring a numeric-valued x-axis always implicitly does: do a 
# 'distinct xxx' and then use the resulting IDs
plt.title("Bitcom Transactions per Merkle Root")
plt.plot(transactions_per_mr.transactions)