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
hacker_news.head("full")
# query to check How many stories are there of each type in the full table

query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type            
        """

# check how big this query will be
hacker_news.estimate_query_size(query)


# Run the query in safe mode 
hacker_types = hacker_news.query_to_pandas_safe(query)
hacker_types
hacker_news.head("comments")
# query to check How many comments have been deleted

query1 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted         
        """

# check how big this query will be
hacker_news.estimate_query_size(query1)


# Run the query in safe mode 
hacker_del = hacker_news.query_to_pandas_safe(query1)
hacker_del
## * **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
   
# query to check max score by type

query2 = """SELECT type, max(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type            
        """

# check how big this query will be
hacker_news.estimate_query_size(query2)

                                                                                 
# Run the query in safe mode 
hacker_score = hacker_news.query_to_pandas_safe(query2)
hacker_score