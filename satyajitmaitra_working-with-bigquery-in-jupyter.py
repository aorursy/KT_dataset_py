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
from google.cloud import bigquery  #syntax for importing bigquery



#you need to connect to bigquery datawarehouse via client 

#this client will look after your connection with bigquery to retrive the data



client = bigquery.Client() 



#creating a reference point to the dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



#API request to fetch the dataset

dataset = client.get_dataset(dataset_ref)



#finding all the tables in the dataset

tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
table_ref = dataset_ref.table("full")



#API request

table = client.get_table(table_ref)
#table is the object that stores details of the table

table.schema
#listing rows to see the data

client.list_rows(table,max_results=5).to_dataframe()
#see the type of schema

type(table.schema)
table.schema[:1]
client.list_rows(table, selected_fields=table.schema[:2],max_results=10).to_dataframe()
Query1="""

SELECT title,url from

`bigquery-public-data.hacker_news.full`

where title != "None" and url !="None"

"""



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)

query_job = client.query(Query1,job_config=safe_config)



Not_null_Titles = query_job.to_dataframe()



Not_null_Titles.head()
Query2="""

SELECT count(text), 'by' as Author from

`bigquery-public-data.hacker_news.full`

group by Author

"""



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)

query_job = client.query(Query2,job_config=safe_config)



Author_per_post = query_job.to_dataframe()

Author_per_post.head()
Query3="""

SELECT count(dead) dead_people, type from 

`bigquery-public-data.hacker_news.full`

group by type

order by 1

"""



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)

query_job = client.query(Query3,job_config=safe_config)



count_wise_dead =query_job.to_dataframe()

count_wise_dead.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))



count_wise_dead.plot.bar()

plt.xlabel("types")

plt.ylabel("count")
