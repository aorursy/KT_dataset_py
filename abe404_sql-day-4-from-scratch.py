import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                             dataset_name='bitcoin_blockchain')
# CTEs help us break up queries into logic parts.

# First part converts integer to timestamp

# Second part gets information on the date of transactions from the timestamp

query = """ with time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) as transactions,
                EXTRACT(MONTH from trans_time) as month,
                EXTRACT(YEAR from trans_time) as year
            FROM time
            GROUP BY year, month
            ORDER BY year, month
        """

# need to expand the safe limit :)
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_month.head(n=13)
# Raw plot
import matplotlib.pyplot as plt
plt.plot(transactions_per_month.transactions)
plt.title('Monthly bitcoin transactions')
# We will need a way to convert our months from integers to names 
# to make the graph more readable.
import calendar
calendar.month_name[12]
import numpy as np

# let's graph the bitcoin transactions per month.
import matplotlib.pyplot as plt

# Change size of graph (se we can see the details)
fig = plt.gcf()
fig.set_size_inches(32, 9)
plt.xlabel('month')
plt.ylabel('No. BitCoin Transactions')

# prepare data for graph
transaction_counts = transactions_per_month.transactions.as_matrix()
month_numbers = transactions_per_month.month.as_matrix()
years = transactions_per_month.year.as_matrix()
month_names = [calendar.month_name[m] for m in month_numbers]
month_year = [m + ' ' + str(y) for m,y in zip(month_names, years)]
x_range = list(range(transaction_counts.shape[0]))
y_range = list(range(0, int(1.1e7), int(1e6)))
y_labels = ["{:,}".format(v) for v in y_range]

# Plot ticks
plt.yticks(y_range, y_labels)
plt.xticks(x_range, month_year, rotation=65)

# Plot grid and real value points
plt.grid()
plt.scatter(x_range, transaction_counts)

# Plot trend line
# calculate a plot the trend line (using simple linear fitting)
# adapted from : http://widu.tumblr.com/post/43624347354/matplotlib-trendline
z = np.polyfit(x_range, transaction_counts, 9)
fitted_line = np.poly1d(z)
plt.plot(x_range, fitted_line(x_range), "r-")

query = """ with time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) as transactions,
                EXTRACT(DAYOFWEEK from trans_time) as day_of_week,
                EXTRACT(YEAR from trans_time) as year
            FROM time
            GROUP BY day_of_week, year
            HAVING year = 2017
            ORDER BY day_of_week
        """

# need to expand the safe limit :)
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day.head(n=10)
days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
plt.xticks(range(len(days)), days)
plt.plot(transactions_per_day.transactions)

query = """ SELECT COUNT(transaction_id) AS transactions,
                merkle_root AS root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY root
        """

# need to expand the safe limit :)
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=41)
# transactions_per_merkle.head()
transactions_per_merkle.sort_values("transactions", ascending=False).head(n=4)
test.head(n=20)
# Lets plot the distribution of transactions counts for each merkle tree (or maybe later)
#import numpy as np
#import pandas as pd
#from scipy import stats, integrate
#import matplotlib.pyplot as plt

# Or we can do the quick and hacky way ;)

# We can see that most of the trees have very few transations
transactions_per_merkle.hist(figsize=(15, 11), bins=50)

import matplotlib.pyplot as plt    
plt.xlabel('transaction_count')
plt.ylabel('tree_count (log scale)')
fig = plt.gcf()
fig.set_size_inches((15, 11))
plt.hist(transactions_per_merkle['transactions'], log=True, bins=200);
query = """ 
            WITH transaction_counts AS
            (
                SELECT COUNT(transaction_id) AS transactions,
                    merkle_root AS root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY root
            )
            SELECT AVG(transactions) as average_transactions
            FROM transaction_counts
        """

# need to expand the safe limit :)
avg_transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=41)
avg_transactions_per_merkle
