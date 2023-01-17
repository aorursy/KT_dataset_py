# import general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import our trusty bigquery helper library
import bq_helper
# create helper
bit_helper = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                     dataset_name = 'bitcoin_blockchain')

# Check out the tables in the space
bit_helper.list_tables()
# Check out the head of one of the tables
bit_helper.head('transactions')
# build a query
query1 = """
        WITH dates AS
        (
        SELECT TIMESTAMP_MILLIS(timestamp) as datetime, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        
        SELECT 
        EXTRACT(DATE from datetime) as date,
        COUNT(transaction_id) as transactions
        FROM dates
        WHERE EXTRACT(YEAR from datetime)=2017
        GROUP BY date
        ORDER BY date
"""

# check out data usage for our query
bit_helper.estimate_query_size(query1)
# run that suckah!
bit_transactions = bit_helper.query_to_pandas_safe(query1, max_gb_scanned=21)
bit_transactions.head(10)
# Let's add a 30-day moving average
# first use the 'date' column as the index
bit_transactions.set_index('date', inplace=True)
bit_transactions['30dMA'] = bit_transactions['transactions'].rolling(30).mean()

plt.style.use('ggplot')
bit_transactions.plot(figsize=(12,8))
plt.title('Bitcoin Transactions Per Day from {} to {}'.format(bit_transactions.index[0], bit_transactions.index[-1]))

min = bit_transactions.transactions.min()
max = bit_transactions.transactions.max()
mindate = bit_transactions.transactions.argmin()
maxdate = bit_transactions.transactions.argmax()

print("The lowest number of transactions was {} which occurred on {}.".format(min, mindate))
print("The highest number of transactions was {} which occurred on {}.".format(max, maxdate))