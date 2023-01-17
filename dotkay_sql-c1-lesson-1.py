# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import bigquery in order to use the API functions

from google.cloud import bigquery
# create a client object

client = bigquery.Client()
dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))

print (len(tables))



# iterate through the tables and print the table id, for eg.

for table in tables:

    print(table.table_id)
full_table_ref = dataset_ref.table("full")

full_table = client.get_table(full_table_ref)
full_table.schema

schema_list = full_table.schema

count = 0

for s in schema_list:

    print (s.field_type)

    if (s.field_type == "STRING"):

        count = count + 1

print (count)
client.list_rows(full_table, max_results=5).to_dataframe()
client.list_rows(full_table, max_results=5, selected_fields=full_table.schema[:1]).to_dataframe()