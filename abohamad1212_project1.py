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

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "world_bank_intl_education" dataset
dataset_ref = client.dataset("chicago_taxi_trips", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "international_education" table
table_ref = dataset_ref.table("taxi_trips")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "international_education" table
df=client.list_rows(table, max_results=5).to_dataframe()
df.info()
# Your code goes here
country_spend_pct_query = """
                          SELECT payment_type,COUNT(unique_key) As Count
                          FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                          GROUP BY payment_type
                          ORDER BY Count DESC
                          LIMIT 5
                          """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
country_spend_pct_query_job = client.query(country_spend_pct_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
country_spending_results = country_spend_pct_query_job.to_dataframe()

# View top few rows of results
print(country_spending_results)
df.info()
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.xlabel("Type of Company")
plt.ylabel("The Percentage of Payment Method")
plt.title("The Percentage of Payment Method by For Most Common Types of Payment")
ax.bar(country_spending_results.payment_type,country_spending_results.Count/country_spending_results.Count.sum())
plt.show()
country_spending_results.Count/country_spending_results.Count.sum()
# Your code goes here
country_spend_pct_query = """
                          SELECT Avg(trip_total) As Avrage_for_all_trip
                          FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                          """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
country_spend_pct_query_job = client.query(country_spend_pct_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
country_spending_results = country_spend_pct_query_job.to_dataframe()

# View top few rows of results
print(country_spending_results)
# Your code goes here
country_spend_pct_query = """
                          SELECT company,Avg(trip_total) As Avrage_for_all_trip
                          FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                          GROUP BY company
                          ORDER BY Avrage_for_all_trip ASC
                          LIMIT 5
                          """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
country_spend_pct_query_job = client.query(country_spend_pct_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
country_spending_results = country_spend_pct_query_job.to_dataframe()

# View top few rows of results
print(country_spending_results)
# Your code goes here
country_spend_pct_query = """
                          SELECT company,Avg(trip_total) As Avrage_for_all_trip
                          FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                          GROUP BY company
                          ORDER BY Avrage_for_all_trip DESC
                          LIMIT 5
                          """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
country_spend_pct_query_job = client.query(country_spend_pct_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
country_spending_results = country_spend_pct_query_job.to_dataframe()

# View top few rows of results
print(country_spending_results)
