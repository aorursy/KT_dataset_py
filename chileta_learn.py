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



# Construct a reference to the "crypto_bitcoin" dataset

dataset_ref = client.dataset("crypto_bitcoin", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "transactions" table

table_ref = dataset_ref.table("transactions")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "transactions" table

client.list_rows(table, max_results=5).to_dataframe()
query_with_CTE = """ 

                 WITH time AS 

                 (

                     SELECT DATE(block_timestamp) AS trans_date

                     FROM `bigquery-public-data.crypto_bitcoin.transactions`

                 )

                 SELECT COUNT(1) AS transactions,

                        trans_date

                 FROM time

                 GROUP BY trans_date

                 ORDER BY trans_date

                 """
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)





query_job = client.query(query_with_CTE, job_config=safe_config)



transactions_by_date = query_job.to_dataframe()



transactions_by_date.head()
transactions_by_date.set_index('trans_date').plot()