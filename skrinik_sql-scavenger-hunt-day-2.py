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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")

# How many stories (use the "id" column) 
#are there of each type (in the "type" column) in the full table?
query1 = """
select type, count(id) as num_type
from `bigquery-public-data.hacker_news.full`
group by type
"""
hacker_news.query_to_pandas_safe(query1)
#How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query2 = """
select sum(id) as total_deleted, deleted
from `bigquery-public-data.hacker_news.comments`
group by deleted
having deleted = True
"""
hacker_news.query_to_pandas_safe(query2)