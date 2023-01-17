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



# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("nhtsa_traffic_fatalities", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "comments" table

table_ref = dataset_ref.table("accident_2015")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "comments" table

client.list_rows(table, max_results=5).to_dataframe()
query = """

        SELECT COUNT(consecutive_number) AS num_accidents,

        EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week,

        EXTRACT(QUARTER FROM timestamp_of_crash) AS quarter

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

        GROUP BY day_of_week, quarter

        ORDER BY quarter, num_accidents DESC

        """
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(query, job_config=safe_config)

accidents_by_day = query_job.to_dataframe()

accidents_by_day