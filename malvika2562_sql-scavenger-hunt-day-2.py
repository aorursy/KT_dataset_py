# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.list_tables()
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


query1 = """ select type,count(id) as count_id
        from `bigquery-public-data.hacker_news.full`
        group by type
"""
type_count = hacker_news.query_to_pandas_safe(query1)

type_count.head()
query2 = """ select count(*)
from `bigquery-public-data.hacker_news.full`
where type = "comment" and deleted = True
"""

comment_count = hacker_news.query_to_pandas_safe(query2)
comment_count.head()
