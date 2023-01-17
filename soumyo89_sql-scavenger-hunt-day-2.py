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
# Solution 1

hacker_news.head("full")
#Solution 1
# query to pass to 
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

diff_types = hacker_news.query_to_pandas_safe(query1)
diff_types.to_csv("diff_types.csv")
# query to pass to 
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'comment'
            GROUP BY deleted
        """

del_comments = hacker_news.query_to_pandas_safe(query2)
del_comments.to_csv("del_comments.csv")