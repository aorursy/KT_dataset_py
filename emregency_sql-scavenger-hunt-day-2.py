# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type            
        """
type_of_stories = hacker_news.query_to_pandas_safe(query)
type_of_stories
# query to pass to 
query = """SELECT deleted, count(id) AS total
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
deleted_comments_count = hacker_news.query_to_pandas_safe(query)
deleted_comments_count
# query to pass to 
query = """SELECT author, avg(ranking) AS avg
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            HAVING avg > 1000
            ORDER BY avg DESC
        """
high_score_authors = hacker_news.query_to_pandas_safe(query)
high_score_authors