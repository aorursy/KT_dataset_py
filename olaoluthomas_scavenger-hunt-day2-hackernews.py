# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="hacker_news")
hacker_news.head("comments")
# tope 5 popular stories
query_1 = """SELECT parent, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10"""

popular_stories = hacker_news.query_to_pandas_safe(query_1)
popular_stories.head()
# count of top 5 comment type
query_2 = """SELECT type, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type"""

type_count = hacker_news.query_to_pandas_safe(query_2)
type_count.head()
# number of deleted comments
query_3 = """SELECT deleted, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True"""

deleted_comments = hacker_news.query_to_pandas_safe(query_3)
deleted_comments
# test of aggregate function