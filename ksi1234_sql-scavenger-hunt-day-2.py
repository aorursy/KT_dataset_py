# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
hacker_news.head("full")
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
querycs = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
count_stories = hacker_news.query_to_pandas_safe(querycs)
count_stories.head()
# How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)
querydc = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = TRUE
        """
deleted_comments = hacker_news.query_to_pandas_safe(querydc)
deleted_comments.head()
# Which authors write the most comments?
queryfq = """SELECT author, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            HAVING COUNT(id) > 1000
            ORDER BY COUNT(id) Desc
        """
frequent_commenters = hacker_news.query_to_pandas_safe(queryfq)
frequent_commenters.head()