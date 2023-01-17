# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "crypto_ethereum" dataset (https://www.kaggle.com/bigquery/ethereum-blockchain)

dataset_ref = client.dataset("crypto_ethereum", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# List all the tables in the "crypto_ethereum" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there's only one!)

for table in tables:  

    print(table.table_id)
BNB_query = """

        SELECT *

        FROM `bigquery-public-data.crypto_ethereum.traces` AS traces

        WHERE traces.to_address = '0xb8c77482e45f1f44de1745f52c74426c631bdd52'

        ORDER BY traces.block_number ASC

        """

BNB_history_query = client.query(BNB_query)

BNB_history = BNB_history_query.to_dataframe()



BNB_history.head()
BNB_history.iloc[-1]
BNB_history.to_csv('outputfile.csv', index=False)
BNB_history.query('transaction_hash =="0x3e68e6e4bb6d018664a5e87906ba35633f99f58beb5dcc20151cfed72b8eae73"')