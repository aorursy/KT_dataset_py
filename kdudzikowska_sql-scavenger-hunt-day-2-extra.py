# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.head("full")
# Your code goes here :)
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query_1 = """SELECT type, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_types = hacker_news.query_to_pandas_safe(query_1)
story_types
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query_2 = """ SELECT deleted, COUNT(*) AS comments
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
            """
deleted_com = hacker_news.query_to_pandas_safe(query_2)
deleted_com

# modify one of the queries you wrote above to use a different aggregate function.
query_3 = """ SELECT type, AVG(score) AS avg_score
    FROM `bigquery-public-data.hacker_news.full`
    WHERE type != 'comment'
    GROUP BY type
    ORDER BY avg_score DESC
    """
avg_score = hacker_news.query_to_pandas_safe(query_3)
avg_score