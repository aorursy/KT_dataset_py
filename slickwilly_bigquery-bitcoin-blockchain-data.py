# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# import plotting library
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = [12.0, 8.0]

# display all outputs within each Juypter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Create helper object for the  the bigQuery data set
blockchain_helper = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                            dataset_name="bitcoin_blockchain")
# inspect the structure
blockchain_helper.list_tables()
# look at a table of the information for both data sets



blockchain_helper.head('transactions')
blockchain_helper.head('blocks')
# lets parse the timestamp data into readable date times
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    block_id
                FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            )
            SELECT COUNT(block_id) AS blocks,
                EXTRACT(DATE FROM trans_time) AS date
            FROM time
            GROUP BY date
            ORDER BY date
        """
blockchain_helper.estimate_query_size(query)
q1_df = blockchain_helper.query_to_pandas(query)
q1_df.head(10)
plt.plot(q1_df['date'] ,q1_df['blocks'])
plt.bar(q1_df['date'], q1_df['blocks'], align='edge')
# when did this outlier occur?
# it looks like there was a large influx of users July 2010:
# https://en.bitcoin.it/wiki/2010#July
q1_df.sort_values('blocks', ascending=False).head(10)
# from this notebook: https://www.kaggle.com/ibadia/bitcoin-101-bitcoin-and-a-useful-insight
# lets find which address has the most number of transactions
QUERY = """
SELECT
    inputs.input_pubkey_base58 AS input_key, count(*)
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    JOIN UNNEST (inputs) AS inputs
WHERE inputs.input_pubkey_base58 IS NOT NULL
GROUP BY inputs.input_pubkey_base58 order by count(*) desc limit 1000
"""
blockchain_helper.estimate_query_size(QUERY)
q2 = blockchain_helper.query_to_pandas(QUERY)
# lets query all transactions this person was involved in
q_input = """
        WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    inputs.input_pubkey_base58 AS input_key,
                    outputs.output_pubkey_base58 AS output_key,
                    outputs.output_satoshis AS satoshis,
                    transaction_id AS trans_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    JOIN UNNEST (inputs) AS inputs
                    JOIN UNNEST (outputs) AS outputs
                WHERE inputs.input_pubkey_base58 = '1NxaBCFQwejSZbQfWcYNwgqML5wWoE3rK4'
            )
        SELECT input_key, output_key, satoshis, trans_id,
            EXTRACT(DATE FROM trans_time) AS date
        FROM time
        --ORDER BY date
        """
blockchain_helper.estimate_query_size(q_input)
q3 = blockchain_helper.query_to_pandas(q_input)
q3.head(10)
# make a datatime type transformation
q3['date'] = pd.to_datetime(q3.date)
q3 = q3.sort_values('date')
# convert satoshis to bitcoin
q3['bitcoin'] = q3['satoshis'].apply(lambda x: float(x/100000000))
print(q3.info())
q3.head(10)
q3.info()
# how many unique addresses are included in this wallets transaction history?
q3['output_key'].nunique()
# how many transactions to the top 10 addresses?
# extreme value for the top address...
q3['output_key'].value_counts().nlargest(10)
# fold difference between the largest and second largest wallet transactions - 44X!
q3['output_key'].value_counts().nlargest(5).iloc[0] / q3['output_key'].value_counts().nlargest(5).iloc[1]
# we should look at transaction activity across time
q3_plot = q3['date'].value_counts()
q3_plot.head()
# plotting params
ax = plt.gca()
ax.scatter(q3_plot.index, q3_plot.values)
#ax.set_yscale('log')
# percentage of unique transactions out of total - the rest must be transactions to multiple addresses
q3['trans_id'].nunique() / len(q3)
q3['output_key'].nunique() / len(q3)
# each trans_id and output_key should have a single occurence if I understand this correctly
# the total amount of these should be the total number of records from the wallet of interest
len(q3.groupby(['trans_id', 'output_key']).nunique()) / len(q3)
# lets plot the value of transactions over time from this wallet
q4_plot = q3.groupby('date', as_index=False)[['bitcoin']].sum()
plt.plot_date(q4_plot['date'], q4_plot['bitcoin'])
q3['month'] = q3['date'].q3.month
q3['year'] = q2['date'].q3.year
q3.head()