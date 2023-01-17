# import package with helper functions 
import bq_helper

# create a helper object for this datasetA
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "commAents" table
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
query1= """SELECT type, COUNT(id) 
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                """
story_type= hacker_news.query_to_pandas_safe(query1)
story_type.head()
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
deleted= hacker_news.query_to_pandas_safe(query2)
deleted.head()
query3 = """SELECT deleted, COUNT(id) as total_deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
deleted_1= hacker_news.query_to_pandas_safe(query3)
deleted_1.head()           
query4 = """SELECT type, COUNT(id) as number_of_stories
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                """
story_type_1= hacker_news.query_to_pandas_safe(query4)
story_type_1.head()
