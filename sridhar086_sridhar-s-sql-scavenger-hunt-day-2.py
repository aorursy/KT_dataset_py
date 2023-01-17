# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table

hacker_news.head("full")
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
query_c = """SELECT type, AVG(descendants) AS avg_desc, AVG(score) AS avg_score, count(id) AS count
            FROM `bigquery-public-data.hacker_news.full` GROUP BY type ORDER BY avg_score DESC
        """
no_of_stories = hacker_news.query_to_pandas_safe(query_c)
no_of_stories
query2 = """SELECT deleted,COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.comments` GROUP BY deleted HAVING deleted=True
        """
deleted_stories = hacker_news.query_to_pandas_safe(query2)
print ( "Number of deleted posts are ",deleted_stories.to_dict()['count'][0])