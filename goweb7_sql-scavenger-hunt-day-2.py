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
query_total = """
              SELECT type AS Type, COUNT(id) AS Total
              FROM `bigquery-public-data.hacker_news.full`
              GROUP BY type
              """

total_df = hacker_news.query_to_pandas_safe(query_total)
print(total_df)

query_deleted = """
                SELECT COUNT(deleted) AS Deleted
                FROM `bigquery-public-data.hacker_news.comments`
                """

deleted_df = hacker_news.query_to_pandas_safe(query_deleted)
print('Deleted comments:', deleted_df.get_value(0, 'Deleted'))

query_alt = """
            SELECT SUM(CAST(deleted AS INT64)) AS Deleted
            FROM `bigquery-public-data.hacker_news.comments`
            """

alt_df = hacker_news.query_to_pandas_safe(query_alt)
print('Deleted comments (alt):', alt_df.get_value(0, 'Deleted'))