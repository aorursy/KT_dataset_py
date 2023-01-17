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
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "openaq" dataset

dataset_ref = client.dataset("openaq", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
table_ref= dataset_ref.table("global_air_quality")

table=client.get_table(table_ref)

client.list_rows(table, max_results=10).to_dataframe()
table.schema
# Query to select all the items from the "city" column where the "country" column is 'US'

query = """

        SELECT city

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
# Set up the query

query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame

us_cities = query_job.to_dataframe()
us_cities
us_cities.city.value_counts().head(5)
query = """

        SELECT city, source_name

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
job = client.query(query)
us_cities = job.to_dataframe()
us_cities