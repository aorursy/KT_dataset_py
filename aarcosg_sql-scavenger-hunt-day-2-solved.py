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
# print the first couple rows of the "full" table
hacker_news.head("full")
# How many stories are there of each type in the full table?
query = """SELECT type AS story_type, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories_by_type = hacker_news.query_to_pandas_safe(query)
stories_by_type.head()
# How many comments have been deleted?
query = """SELECT COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
comments_deleted = hacker_news.query_to_pandas_safe(query)
print('{} comments have been deleted'.format(comments_deleted['count'][0]))
# [Extra] How many deleted stories are there of each type in the full table with less than 500 deleted stories
query = """SELECT type AS story_type, COUNTIF(deleted=True) AS count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING COUNTIF(deleted=True) < 500
        """
stories = hacker_news.query_to_pandas_safe(query)
stories.head()