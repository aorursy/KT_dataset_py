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
hacker_news.head("full")
query3 = """SELECT type, COUNT(id) AS sotry_count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_type = hacker_news.query_to_pandas_safe(query3)
story_type
query4 = """SELECT deleted, count(id) AS deleted_comment
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            GROUP BY deleted
        """
deleted_comment = hacker_news.query_to_pandas_safe(query4)
deleted_comment
query5 = """SELECT type, Avg(score) AS deleted_comment
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
avg_score = hacker_news.query_to_pandas(query5)
avg_score
