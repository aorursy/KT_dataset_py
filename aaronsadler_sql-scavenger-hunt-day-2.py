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
# The number of stories that use the id column
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
#SQL code question 1
query_1 = """select type 
            , count(distinct id) as story_count
            from `bigquery-public-data.hacker_news.full_201510`
            where type = 'story'
            group by type
            
"""
use_col_id = hacker_news.query_to_pandas_safe(query_1)

#Outputs Q1:
use_col_id
#SQL code question 2
query_2 = """select count(id) as deleted_comments
            from `bigquery-public-data.hacker_news.comments`
            where deleted = TRUE
"""
deleted_comments = hacker_news.query_to_pandas_safe(query_2)

#Outputs Q1:
deleted_comments
#SQL Option extra credit
query_3 = """select count(id) as deleted_comments
                , sum(ranking) as sum_rank
            from `bigquery-public-data.hacker_news.comments`
            where deleted = TRUE
"""
deleted_comments_with_rank = hacker_news.query_to_pandas_safe(query_3)

#Outputs Q1:
deleted_comments_with_rank