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
myqy0 = """SELECT COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
        """
df0 = hacker_news.query_to_pandas_safe(myqy0)
df0.head()
# Your code goes here :)
myqy1 = """SELECT COUNT(id), type
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
        """
df1 = hacker_news.query_to_pandas_safe(myqy1)
df1.head()
myqy2 = """SELECT COUNT(id), deleted
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY deleted
        """
df2 = hacker_news.query_to_pandas_safe(myqy2)
df2.head()
myqy2b = """SELECT COUNT(id), deleted
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
         """
df2b = hacker_news.query_to_pandas_safe(myqy2b)
df2b.head()