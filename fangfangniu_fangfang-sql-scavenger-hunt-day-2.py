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
# import the bq_helper package
import bq_helper
# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("full")
# preview the first couple of rows of the "full" table
hacker_news.head("full")
# this query looks at the rows where the type is story in the full table in the hacker_news
# dataset, and count the rows which share the same text.
query_unique_stories_count = """SELECT text, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'story'
            GROUP BY text """

# check how big this query will be
hacker_news.estimate_query_size(query_unique_stories_count)
# check out the number of times each unique story is reused
unique_stories_count = hacker_news.query_to_pandas(query_unique_stories_count)
unique_stories_count.head()
# the length of the new dataframe is the total number of unique stories in the full table 
len(unique_stories_count)
# the average times that a unique story is reused/reappeared
unique_stories_count.f0_.mean()
# the max times that a unique story is reused/reappeared
unique_stories_count.f0_.max()
query_simple = """SELECT COUNT(text) 
             FROM `bigquery-public-data.hacker_news.full` 
             WHERE type = 'story' 
             GROUP BY text """
# check how big this query will be
hacker_news.estimate_query_size(query_simple)
# the length of the new dataframe is the total number of unique stories in the full table 
unique_stories_count_simple = hacker_news.query_to_pandas(query_simple)
len(unique_stories_count_simple)
query_distinct = """ SELECT COUNT(DISTINCT text) 
             FROM `bigquery-public-data.hacker_news.full` 
             WHERE type = 'story' """
# check how big this query will be
hacker_news.estimate_query_size(query_distinct)
# the value in the f0_ column is the total number of unique stories in the full table 
unique_stories_count_distinct = hacker_news.query_to_pandas(query_distinct)
unique_stories_count_distinct.head()
# preview the first couple of rows of the "comments" table
hacker_news.head("comments")
# preview the first ten entries in the deleted column of the comments table
hacker_news.head("comments", selected_columns="deleted", num_rows=10)
# this query looks in the comments table in the hacker_news
# dataset, then count the rows which are deleted.
query_comments_deleted = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            where deleted = True """

# check how big this query will be
hacker_news.estimate_query_size(query_comments_deleted)
# check out the number of comments that are deleted 
num_comments_deleted = hacker_news.query_to_pandas(query_comments_deleted)
num_comments_deleted.head()
# this query looks in the comments table in the hacker_news
# dataset, then count the rows which are deleted.
query_having = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True """

# check how big this query will be
hacker_news.estimate_query_size(query_having)
# check out the number of comments that are deleted 
num_comments_deleted_having = hacker_news.query_to_pandas(query_having)
num_comments_deleted_having.head()