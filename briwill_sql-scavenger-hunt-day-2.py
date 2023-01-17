# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# query to pass to 
query = """SELECT type, COUNT(id) as cnt
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY cnt
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
story_types_df = hacker_news.query_to_pandas_safe(query)

story_types_df
# query to pass to 
query2 = """SELECT deleted, COUNT(id) as cnt
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            GROUP BY deleted
        """

deleted_comments_df = hacker_news.query_to_pandas_safe(query2)

deleted_comments_df
# User IDs for the of the 10 longest comments
query3 = """SELECT FORMAT_TIMESTAMP("%m", time_ts) as month, AVG(LENGTH(text)) as avg
            FROM `bigquery-public-data.hacker_news.stories`
            GROUP BY month
            HAVING month <> 'None'
            ORDER by month
          """
avg_story_length_df = hacker_news.query_to_pandas_safe(query3)

avg_story_length_df