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
#Creating a client object

client=bigquery.Client()

#constructing a reference to dataset

dataset_ref=client.dataset("hacker_news",project="bigquery-public-data")

#API request_fetching the dataset

dataset=client.get_dataset(dataset_ref)

#construct a reference to the comments table

table_ref=dataset_ref.table("comments")

#API request_fetching the table

table=client.get_table(table_ref)

#Preview the first ten lines of the comments table

client.list_rows(table,max_results=10).to_dataframe()

#Query to select comments that received more than n number of replies

query_popular="""

              SELECT parent,COUNT(id)

              FROM `bigquery-public-data.hacker_news.comments`

              GROUP BY parent

              HAVING COUNT(id)>10

              """

#to rename column resulting form id name use "as newcommentname" after COUNT(id) - Aliasing
# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_popular, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

popular_comments = query_job.to_dataframe()



# Print the first five rows of the DataFrame

popular_comments.head()