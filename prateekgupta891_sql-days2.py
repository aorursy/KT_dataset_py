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



dataset_ref = client.dataset("openaq",project="bigquery-public-data")



dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
#q1

query = """

select city 

from `bigquery-public-data.openaq.global_air_quality`

where country='US'

"""

#client = bigquery.Client()

query_job = client.query(query)



us_cities = query_job.to_dataframe()

print(us_cities.city.value_counts())
#q2

query = """

select city,country

from `bigquery-public-data.openaq.global_air_quality`

where country = 'US'

"""

query_job = client.query(query)

table = query_job.to_dataframe()

print(table.head())
#q3

query = """

select *

from `bigquery-public-data.openaq.global_air_quality`

where country = 'US'

"""

query_job = client.query(query)

table = query_job.to_dataframe()

table.head()
#q4

from google.cloud import bigquery

query = """

select score,title

from `bigquery-public-data.hacker_news.full`

where type = "job"

"""

#dataset_ref = client.dataset('hacker_news',project ='bigquery-public-data')

client= bigquery.Client()

#to estimate the time of the query before running it

dry_run_config = bigquery.QueryJobConfig(dry_run = True)

dry_run_query_job = client.query(query,job_config = dry_run_config) #api call



print("The query will process in {} bytes.".format(dry_run_query_job.total_bytes_processed))

#run the query to run with 100mb of data

ONE_MB = 1000*1000

ONE_HUNDRED_MB = 100*ONE_MB

safe_config = bigquery.QueryJobConfig(maximum_bytes_billded = ONE_HUNDRED_MB)

safe_query_job = client.query(query,job_config=safe_config)

table = safe_query_job.to_dataframe()

table.score.mean()