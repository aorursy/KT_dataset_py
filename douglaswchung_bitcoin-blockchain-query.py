# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # helper functions for putting BigQuery results in Pandas DataFrames
from matplotlib import pyplot as plt # plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# create a helper object for our bigquery dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "bitcoin_blockchain")
# print a list of all the tables in the hacker_news dataset
bitcoin_blockchain.list_tables()
# print information on all the columns in the "blocks" table
# in the bitcoin_blockchain dataset
bitcoin_blockchain.table_schema("transactions")
# preview the first couple lines of the "blocks" table
bitcoin_blockchain.head("transactions")
# preview the first fuve entries in the by column of the blocks table
bitcoin_blockchain.head("transactions", selected_columns="block_id", num_rows=5)
# this query looks in the blocks table in the bitcoin_blockchain
# dataset, then gets the block_id column from every row where 
# the timestamp column is greater than 1288778654000.
query = """SELECT version
            FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            WHERE timestamp > 1515949679000; """

# check how big this query will be
bitcoin_blockchain.estimate_query_size(query)
# only run this query if it's less than 100 MB
recent_version = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=0.1)
plt.hist(recent_version)