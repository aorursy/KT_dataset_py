# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import bq_helper
# Any results you write to the current directory are saved as output.
hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", dataset_name = "hacker_news")
hacker_news.list_tables()
hacker_news.table_schema("full")
hacker_news.head("full")
hacker_news.head("stories")
hacker_news.head("comments")
hacker_news.head("full", selected_columns = "by", num_rows = 10)
hacker_news.head("stories", selected_columns = "url", num_rows = 20)
hacker_news.head("comments", selected_columns = "text", num_rows = 5)
query_full = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

hacker_news.estimate_query_size(query_full)

query_comments = """SELECT author
                 FROM `bigquery-public-data.hacker_news.comments`
                 WHERE ranking = 1 """

hacker_news.estimate_query_size(query_comments)

query_stories = """SELECT url
                 FROM `bigquery-public-data.hacker_news.stories`
                 WHERE dead = True """

hacker_news.estimate_query_size(query_stories)
hacker_news.query_to_pandas_safe(query, max_gb_scanned = 0.1)

job_post_scores = hacker_news.query_to_pandas_safe(query)
job_post_scores.head()
job_post_scores.mean()
job_post_scores.score.mean()
job_post_scores.score.mean()