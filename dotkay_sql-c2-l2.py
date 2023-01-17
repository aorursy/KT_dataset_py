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
# import bigquery

from google.cloud import bigquery



# create a client object

client = bigquery.Client()
# create a reference to the dataset

dataset_ref = client.dataset('crypto_bitcoin', project='bigquery-public-data')



# fetch the data using the dataset_ref

dataset = client.get_dataset(dataset_ref)



# view the interesting info in the dataset, like the tables present

tables = list(client.list_tables(dataset))

for table in tables:

    print (table.table_id)

    
# view the top rows of the transactions table



# create a reference to transactions table

transactions_table_ref = dataset_ref.table('transactions')



# get the table

transactions_table = client.get_table(transactions_table_ref)



# view some rows of the table to see what kind of data is present

client.list_rows(transactions_table, max_results=5).to_dataframe()
cte_query = """

WITH time AS

(

    SELECT DATE(block_timestamp) AS t_date

    FROM `bigquery-public-data.crypto_bitcoin.transactions`

)

SELECT COUNT(1) AS transactions, t_date

FROM time

GROUP BY t_date

ORDER BY t_date

"""



# set the query configs

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)



# issue the query

cte_query_job = client.query(cte_query, job_config=safe_config)



# convert the result to a dataframe so that we can view it

transactions_by_day = cte_query_job.to_dataframe()



# view the results

transactions_by_day.head(10)
# let's plot it

transactions_by_day.set_index('t_date').plot()