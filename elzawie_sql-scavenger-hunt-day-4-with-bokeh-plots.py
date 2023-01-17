# Import package with helper functions 
import bq_helper

# Create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# Print the Bitcoin Blockchain dataset's schema
bitcoin_blockchain.list_tables()
# Display the first 5 rows of 'transactions' table 
bitcoin_blockchain.head('transactions')
query_1 = """ WITH timetable AS
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, 
                     transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
            SELECT COUNT(transaction_id) AS NumOfTransactions,
                EXTRACT(DAY FROM transaction_time) AS Day,
                EXTRACT(MONTH FROM transaction_time) AS Month,
                EXTRACT(YEAR FROM transaction_time) AS Year
            FROM timetable
            WHERE EXTRACT(YEAR FROM transaction_time) = 2017
            GROUP BY year,month,day
            ORDER BY NumOfTransactions DESC
          """
# Estimate query size
bitcoin_blockchain.estimate_query_size(query_1)
# Note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_1, max_gb_scanned=21)
# Print first 5 rows
transactions_per_day.head(5)
# Import pandas library and create new 'Date' column which we'll use for plotting purposes
import pandas as pd
transactions_per_day['Date'] = pd.to_datetime(transactions_per_day[['Year','Month','Day']]) 

# Display first 5 rows
transactions_per_day.head()
# Import Bokeh plotting library
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import NumeralTickFormatter
output_notebook()
# Initialize ColumnDataSource object
source_1 = ColumnDataSource(data=dict(
    dats = transactions_per_day['Date'][:50],
    nums = transactions_per_day['NumOfTransactions'][:50]
))

# Set plot properties
daily_plot = figure(x_axis_label = "Date", y_axis_label = "Number of Transactions", 
                    x_axis_type='datetime', tools = "pan,box_zoom,reset", plot_width = 800,
                    plot_height = 500, title="Transactions Per Day")

daily_plot.circle('dats', 'nums', size = 5, color = '#FFD447', source = source_1)
daily_plot.yaxis[0].formatter = NumeralTickFormatter(format="0,0")
show(daily_plot)
query_2 = """  WITH timetable AS
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, 
                     transaction_id, 
                     merkle_root AS MerkleTree
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
            SELECT COUNT(transaction_id) AS NumOfTransactions, MerkleTree,
                EXTRACT(DAY FROM transaction_time) AS Day,
                EXTRACT(MONTH FROM transaction_time) AS Month,
                EXTRACT(YEAR FROM transaction_time) AS Year
            FROM timetable
            GROUP BY MerkleTree, Year, Month, Day
            ORDER BY NumOfTransactions DESC
          """
# Estimate query size
bitcoin_blockchain.estimate_query_size(query_2)
# Note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query_2, max_gb_scanned=40)
transactions_per_merkle.head(10)
# Initialize ColumnDataSource object
source_2 = ColumnDataSource(data=dict(
    index = list(transactions_per_merkle.index.values)[:100],
    nums = transactions_per_merkle['NumOfTransactions'][:100],
))

# Set plot properties
merkle_plot = figure(x_axis_label = "Merkle index", y_axis_label = "Number of Transactions",
                   tools = "pan,box_zoom,reset", plot_width = 800,
                   plot_height = 500, title = "Transactions Per Merkle")

merkle_plot.circle(x = 'index' ,y = 'nums', size = 5, color = '#30FF49', source = source_2)
merkle_plot.yaxis[0].formatter = NumeralTickFormatter(format="0,0")
show(merkle_plot)