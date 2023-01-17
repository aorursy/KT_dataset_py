# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")

# How many stories (use the "id" column) are there of each type 
#(in the "type" column) in the full table?
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)

popular_stories.head()
# print the first couple rows of the "comment" table
hacker_news.head("comments")

# How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
hacker_news.query_to_pandas_safe(query)
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

# print the first couple rows of the "full" table
hacker_news.head("full")

# Find the story type and sum of scores for each story type where number of stories is less than 12000.
query = """SELECT type, SUM(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type 
            HAVING COUNT(id)< 12000
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)

popular_stories.head()