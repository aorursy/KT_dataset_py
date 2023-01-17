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
hacker_news.head("full")
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING type = "story"
        """
count_stories = hacker_news.query_to_pandas_safe(query1)
count_stories
query2 = """
            SELECT deleted,COUNT(id)
                    FROM `bigquery-public-data.hacker_news.comments`
                    GROUP BY deleted
                    having deleted = True
        """
count_deleted_comment = hacker_news.query_to_pandas_safe(query2)
count_deleted_comment
query3 = """
            SELECT parent,MAX(time)
                    FROM `bigquery-public-data.hacker_news.comments`
                    where parent = 2841589 or parent = 38251 or parent = 4538227
                    GROUP BY parent
        """
new_comment = hacker_news.query_to_pandas_safe(query3)
new_comment
# hacker_news.head("comments")
