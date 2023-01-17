# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Any results you write to the current directory are saved as output.
import bq_helper

# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")

hacker_news.table_schema('full')

#list unique story ids
query = """ SELECT COUNT(DISTINCT id)  from `bigquery-public-data.hacker_news.full` """
distinct_story_df= hacker_news.query_to_pandas_safe(query)
distinct_story_counts = distinct_story_df.iloc[0,0]
print (distinct_story_counts)
query = """ SELECT COUNT(deleted) from `bigquery-public-data.hacker_news.full` WHERE deleted=TRUE """
deleted_comments_df = hacker_news.query_to_pandas_safe(query)
deleted_comments_count = deleted_comments_df.iloc[0,0]
print (deleted_comments_count)