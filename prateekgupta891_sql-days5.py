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



client = bigquery.Client()



dataset_ref = client.dataset("crypto_bitcoin",project="bigquery-public-data")

dataset= client.get_dataset(dataset_ref)



table_ref = dataset_ref.table('transactions')

table= client.get_table(table_ref)



client.list_rows(table,max_results = 5).to_dataframe()
query_with_CTE = """ 

with time as 

(

select date(block_timestamp) as trans_date

from `bigquery-public-data.crypto_bitcoin.transactions`

)

select count(1) as transactions,trans_date

from time

group by trans_date

order by trans_date

"""



safe_config = bigquery.QueryJobConfig(maximum_bytes_billied=10**10)

query_job = client.query(query_with_CTE,job_config = safe_config)



transaction_by_date = query_job.to_dataframe()



print(transaction_by_date.iloc[:100])
#plotting the bitcoin transaction data



transaction_by_date.set_index('trans_date').plot(figsize=(12,8))