# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
query = """ SELECT parent, COUNT(ID) as Comment_Counts
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING Comment_Counts > 10
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
hacker_news.head("full")
query = """SELECT type, COUNT(id) AS num_of_story
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
types = hacker_news.query_to_pandas_safe(query)
types
hacker_news.head("comments")
query = """SELECT deleted, COUNT(id) as amount
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
            """
deleted = hacker_news.query_to_pandas_safe(query)
deleted
type(deleted)
query = """SELECT COUNT(deleted)
            FROM ( SELECT deleted FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True )
            """
total_count = hacker_news.query_to_pandas_safe(query)
total_count.iloc[0, 0]
query = """SELECT COUNTIF(deleted = True)
            FROM `bigquery-public-data.hacker_news.comments`
            """
total_count = hacker_news.query_to_pandas_safe(query)
total_count