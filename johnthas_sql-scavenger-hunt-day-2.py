# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # BigQuery
hacker_news = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='hacker_news')
hacker_news.head('full')
query = """
SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""

# check how big this query will be
hacker_news.estimate_query_size(query)
id_df = hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.5)
id_df.head()
query = """
SELECT deleted, count(deleted)
FROM `bigquery-public-data.hacker_news.full`
WHERE type = 'comment'
GROUP BY deleted
"""

# check how big this query will be
hacker_news.estimate_query_size(query)
id_df = hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.5)
id_df.head()