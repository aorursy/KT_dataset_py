# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head('transactions')
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
transactions_per_month.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)
daily_query = """ WITH daily_transactions AS
                             (
                                 SELECT TIMESTAMP_MILLIS(timestamp) AS time,
                                        transaction_id
                                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                             )
                             SELECT EXTRACT(DAY FROM time) AS day,
                                    EXTRACT(MONTH FROM time) AS month,
                                    EXTRACT(YEAR FROM time) AS year,
                                    COUNT(transaction_id) AS transactions_total
                             FROM daily_transactions
                             GROUP BY day, month, year
                             HAVING year = 2017
                             ORDER BY month, day"""
daily_transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(daily_query, max_gb_scanned=21)
from datetime import datetime
daily_transactions_2017['date'] = daily_transactions_2017.apply(lambda row: datetime(
                              row['year'], row['month'], row['day']), axis=1)
daily_transactions_2017.head()
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode() 
data = [go.Scatter(x=daily_transactions_2017.date, y=daily_transactions_2017.transactions_total)]
layout= go.Layout(autosize= True, title= 'Daily Bitcoin transactions of 2017',
       yaxis=dict( title= 'Total transactions '))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
merkle_query = """ WITH merkle AS
                   (
                       SELECT merkle_root,
                              transaction_id
                       FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                   )
                   SELECT COUNT(transaction_id) AS transactions_number, merkle_root
                   FROM merkle
                   GROUP BY merkle_root
                   ORDER BY transactions_number DESC
                """
merkle_transactions = bitcoin_blockchain.query_to_pandas_safe(merkle_query, max_gb_scanned=37)
merkle_transactions.head()