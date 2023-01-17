# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# Any results you write to the current directory are saved as output.
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
hacker_news.list_tables()
hacker_news.table_schema("full")
hacker_news.head("full")
#Question1
# How many stories are there of each type?
query_1 = """SELECT type, COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY type"""
count_of_types = hacker_news.query_to_pandas_safe(query_1)
count_of_types.head()
count_of_types.shape
# Quetion2
# How many comments have been deleted?
query2 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = TRUE
         """
count_of_deleted_comments = hacker_news.query_to_pandas_safe(query2)
count_of_deleted_comments.head()
# Question 3: use aggregate function
query3 = """SELECT AVG(ranking), parent
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
         """
average_ranking_of_comments = hacker_news.query_to_pandas_safe(query3)
average_ranking_of_comments.head()
average_ranking_of_comments.shape
  