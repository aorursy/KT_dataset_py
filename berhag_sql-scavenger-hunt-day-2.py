import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                       dataset_name ='hacker_news')
hacker_news.list_tables()
hacker_news.head('full')
query = """ SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
hacker_news.estimate_query_size(query)
df = hacker_news.query_to_pandas_safe(query)
df.head()
query = """SELECT deleted, count(deleted)
           FROM `bigquery-public-data.hacker_news.full`
           WHERE type = 'comment'
           GROUP BY deleted
           """
hacker_news.estimate_query_size(query)
df = hacker_news.query_to_pandas_safe(query)
df.head()
