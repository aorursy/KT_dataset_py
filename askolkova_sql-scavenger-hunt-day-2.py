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
hacker_news.list_tables()
hacker_news.head('full')
query2 = """
        SELECT type, COUNT(time)
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        """
stories_type = hacker_news.query_to_pandas_safe(query2)
stories_type.head()
query3 = """
        SELECT COUNT(id)
        FROM `bigquery-public-data.hacker_news.comments`
        WHERE deleted = True 
        """
commenta_del = hacker_news.query_to_pandas_safe(query3)
commenta_del.head()
query4 = """
        SELECT MAX(id)
        FROM `bigquery-public-data.hacker_news.comments`
        WHERE deleted = True 
        """
commenta_del = hacker_news.query_to_pandas_safe(query4)
commenta_del.head()