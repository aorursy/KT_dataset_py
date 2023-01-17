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



dataset_ref = client.dataset('nhtsa_traffic_fatalities',project='bigquery-public-data')

dataset = client.get_dataset(dataset_ref)



table_ref = dataset_ref.table("accident_2015")

table = client.get_table(table_ref)



client.list_rows(table,max_results=5).to_dataframe()
query = """

select count(consecutive_number) as num_accidents,extract(dayofweek from timestamp_of_crash) as day_of_week

from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

group by day_of_week

order by day_of_week desc

"""



safe_config = bigquery.QueryJobConfig(maximum_bytes_billied=10**8)

query_job = client.query(query,job_config = safe_config)



accidents_by_day = query_job.to_dataframe()



accidents_by_day