
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


import bq_helper
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

hacker_news = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                      dataset_name='hacker_news')
# check data set schema
print(hacker_news.list_tables())

# check schema details on a table
print(hacker_news.table_schema('comments'))

# print few columns of a table
print(hacker_news.head('comments'))

print(hacker_news.head('comments', selected_columns='by',num_rows=10))
# Check the size of a query
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

print(hacker_news.estimate_query_size(query))
# run a query under limit
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
# score the dataframe returned from the query and check mean()
job_type_scores=hacker_news.query_to_pandas_safe(query)

print(job_type_scores.mean())
# save result to a csv
job_type_scores.to_csv('job_post_scores.csv')