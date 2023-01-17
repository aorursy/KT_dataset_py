# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bq

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input director


hackernews=bq.BigQueryHelper(active_project="bigquery-public-data",dataset_name="hacker_news")
hackernews.list_tables()

hackernews.head("comments")
hackernews.head("full")
# Scavenger hunt:
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

query1="""SELECT type, COUNT(id) as total_count FROM `bigquery-public-data.hacker_news.full` GROUP BY type"""

number_stories= hackernews.query_to_pandas_safe(query1)
number_stories
# Scavenger hunt: 
# How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query2 = """SELECT deleted,COUNT(id) as count_deleted_comments 
            FROM `bigquery-public-data.hacker_news.comments` 
            WHERE deleted=True
            GROUP BY deleted """
hackernews.estimate_query_size(query2)
comments_deleted =hackernews.query_to_pandas_safe(query2)
comments_deleted