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

client = bigquery.Client()

dataset_ref = client.dataset("nhtsa_traffic_fatalities", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)

table_ref = dataset_ref.table("accident_2015")

table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()
# Query to find out the number of accidents for each day of the week

query = """

        SELECT COUNT(consecutive_number) AS num_accidents, 

               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

        GROUP BY day_of_week

        ORDER BY num_accidents DESC

        """
type(query)
# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

accidents_by_day = query_job.to_dataframe()
accidents_by_day