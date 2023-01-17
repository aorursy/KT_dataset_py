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



dataset_ref  = client.dataset("hacker_news",project="bigquery-public-data")



dataset = client.get_dataset(dataset_ref)



table_ref = dataset_ref.table("comments")



table = client.get_table(table_ref)



client.list_rows(table,max_results=5).to_dataframe()
query_popular = """

                SELECT parent, COUNT(id)

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

                """
safe_config = bigquery.QueryJobConfig(maximum_byte_billed = 10**10)

query_job = client.query(query_popular,job_config=safe_config)

popular_comments = query_job.to_dataframe()



popular_comments.head()

query_popular = """

                SELECT parent, COUNT(id) AS NumPosts

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

                """



safe_config = bigquery.QueryJobConfig(maximum_byte_billed = 10**10)

query_job = client.query(query_popular,job_config=safe_config)

improved_df = query_job.to_dataframe()



improved_df.head()