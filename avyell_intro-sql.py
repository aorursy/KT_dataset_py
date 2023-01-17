# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from google.cloud import bigquery
# to create a client object:
# note that i don't have a bigquery connection with google
# so detached my kaggle google association so that i could use kaggle's default one
# grab the data set before running this block
client = bigquery.Client()
# construct a reference to the data set
dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')

# API request to fetch the data set
dataset = client.get_dataset(dataset_ref)
# list the tables in the dataset
# will need a for loop to list the ids for each table

tables = list(client.list_tables(dataset))

for table in tables:
    print(table.table_id)
# we can also fetch a table, the same way we fetched a data set
# construct a reference to the table
table_ref = dataset_ref.table('full')

#API request to fetch the table
table = client.get_table(table_ref)
table.schema
# the table is the reference we just pulled
# we want to list out the first 5 rows of the table.
# the list_rows method is the one that actually lists out the data
# the to_dataframe() method is the one that makes it pandas friendly
client.list_rows(table, max_results=5).to_dataframe()
# if we want to see a specific column:
# add the parameter selected_fields=table.schema[:1] (first col)
client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()
