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
# Your code goes here :)
query = """
    select type, count(id)
    from `bigquery-public-data.hacker_news.full`
    group by type
    """
story_count_by_type = hacker_news.query_to_pandas_safe(query)
story_count_by_type
query = """
    select count(*) as deleted_comments_count
    from `bigquery-public-data.hacker_news.comments`
    where deleted = True
"""
hacker_news.query_to_pandas_safe(query)

query = """
select avg(comment_count)
from(
    select parent, count(id) as comment_count
            from `bigquery-public-data.hacker_news.full`
            group by parent
    )
     as average_comment_count
"""
hacker_news.query_to_pandas_safe(query)

x