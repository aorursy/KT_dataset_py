# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from google.cloud import bigquery
client = bigquery.Client()
# in bighquery each dataset is contained in soe project
# for example "Hacker_news" data set is contained in "bigquery-public-data"
# we will tell our client to get the data from the project by using the dataset method and we will store it in our variable
dataset_ref = client.dataset("hacker_news", project = "bigquery-public-data")
# now that our client has fulfilled our request in bringing the dataset from projct to the dataset referece
# now our client is sitting ideal, so we will request our client to pull the data from the reference to our dataset variable
# using get_dataset() method

dataset = client.get_dataset(dataset_ref)
# now our data is loaded into dataset variable
# every dataset is the collection of tables, think it as a spreadsheet file containing multiple tables all composed of 
# rows and columns.
# list all the tables present in the dataset usint list_tables()  method
# remember our client is sitting ideal so we will again order him to list the tables
tables = list(client.list_tables(dataset))
# to print the table id present
for i in tables:
    print(i.table_id)
# Similar to how we fetched a dataset, we can fetch a table. In the code cell below, we fetch the full table in 
# the hacker_news dataset.
# remember that our client will bring every information from the reference dataset named "dataset_ref" as it has the 
# reference to the original dataset present in the "bigquery-public-dataset"
table_ref = dataset_ref.table('full')

# now fetch the table from the reference 
table = client.get_table(table_ref)
# to view the schema of the table 'full'
table.schema
# We can use the list_rows() method to check just the first five lines of of the full table to make sure this is right.
# (Sometimes databases have outdated descriptions, so it's good to check.) 
# This returns a BigQuery RowIterator object that can quickly be converted to a pandas DataFrame with the to_dataframe() method.
client.list_rows(table, max_results = 5).to_dataframe()
# Preview the first five entries in the "by" column of the "full" table
# we can also select a particular colums to look its values
# here we are using slicing to display only the 0th column as upper index is excluded in python list slicing
client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()
# Preview the first five entries in the "by" column of the "full" table
# looking at only 2 columns
client.list_rows(table, selected_fields=table.schema[:2], max_results=5).to_dataframe()
# looking only last column
# Preview the first five entries in the "by" column of the "full" table
client.list_rows(table, selected_fields=table.schema[-1:], max_results=5).to_dataframe()
