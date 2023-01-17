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
#how many stories in the full table
query = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
           """


# safe query 
stories_number = hacker_news.query_to_pandas_safe(query) 
#print result
stories_number.head()
#deleted comments
query1 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            Group BY deleted
           
            
            """ 
#safe query 
deleted1 = hacker_news.query_to_pandas_safe(query1)

#print result
deleted1
#trying CAST
query2 = """SELECT SUM(CAST(deleted as INT64)) AS Deleted
            FROM `bigquery-public-data.hacker_news.comments` 
            """
#safe query 
deleted2 = hacker_news.query_to_pandas_safe(query2)

#print result
deleted2 [['Deleted']]
