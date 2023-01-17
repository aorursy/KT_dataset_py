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
#Building a query
query1 = """SELECT type, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY count DESC
            """
#Saving query as data frame
types_count = hacker_news.query_to_pandas_safe(query1)
#Head of a data frame
types_count.head()
#Looks like commenting news is very popular
#Making the same procedure
query2 = """SELECT COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted is True
            """
deleted = hacker_news.query_to_pandas_safe(query2)
deleted
#Adding new column to our first query 
query3 = """SELECT type, AVG(score) as average, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY count DESC
            """
modify = hacker_news.query_to_pandas_safe(query3)
modify.head()
#As we can see all comments have NaN value in score