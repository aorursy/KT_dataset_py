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
# Set your own project id here

# PROJECT_ID = 'your-google-cloud-project'

# from google.cloud import bigquery

# bigquery_client = bigquery.Client(project=PROJECT_ID)
from google.cloud import bigquery
# creating client Hacker news project

client = bigquery.Client(project="bigquery-public-data")
# ref to hecker_news dataset

dataset_ref = client.dataset("hacker_news")
# fetching the data - api request

dataset = client.get_dataset(dataset_ref)
# list all tables in dataset

tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
# constructing a ref to full table

table_ref = dataset_ref.table("full")



#fetching table

table = client.get_table(table_ref)
# get schema for table

table.schema
# get the rows

client.list_rows(table, max_results = 10)

# gets us a RowIterator object
client.list_rows(table, max_results = 10).to_dataframe()
# to get information for individual field/column

client.list_rows(table,selected_fields=table.schema[:1], max_results = 10).to_dataframe()