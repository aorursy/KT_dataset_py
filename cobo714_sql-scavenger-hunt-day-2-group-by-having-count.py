# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="hacker_news")

hacker_news.head("comments")
query = """SELECT parent, COUNT(id)
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY parent
           HAVING COUNT(id) > 10
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
hacker_news.head("full")
scav1 = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
        """

story_types = hacker_news.query_to_pandas_safe(scav1)
story_types
hacker_news.head("comments")
scav2 = """SELECT COUNT(id)
           FROM `bigquery-public-data.hacker_news.comments`
           WHERE deleted = True
        """

deleted_comments = hacker_news.query_to_pandas_safe(scav2)
deleted_comments
EC1 = """SELECT `by`, COUNT(id) AS num
         FROM `bigquery-public-data.hacker_news.full`
         WHERE `by` != ""
         GROUP BY `by`
         HAVING COUNT(id) > 12500
      """

top_contributors = hacker_news.query_to_pandas_safe(EC1)
top_contributors[:10]