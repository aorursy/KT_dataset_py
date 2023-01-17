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
# Counts by Type
query = """SELECT type, COUNT(id) as type_count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

type_counts = hacker_news.query_to_pandas_safe(query)
type_counts

# Deleted Comments
query = """SELECT COUNT(id) as deleted_count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = TRUE
        """

deleted = hacker_news.query_to_pandas_safe(query)
deleted
# Average score by Type
query = """SELECT type, AVG(score) as type_count
            FROM `bigquery-public-data.hacker_news.full`
            WHERE not score is null
            GROUP BY type
        """

type_avg_score = hacker_news.query_to_pandas_safe(query)
type_avg_score