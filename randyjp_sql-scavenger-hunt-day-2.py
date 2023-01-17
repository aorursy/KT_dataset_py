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
## 1) How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query2 = """SELECT type, COUNT(id) AS Number_Stories
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

stories_type_number = hacker_news.query_to_pandas_safe(query2)
stories_type_number.head()
##  How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query3 = """SELECT COUNT(id) as Deleted
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted is TRUE
        """

deleted_comments = hacker_news.query_to_pandas_safe(query3)
deleted_comments.head()