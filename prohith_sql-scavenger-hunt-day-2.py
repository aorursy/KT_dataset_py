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
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
stories_query = """
                    SELECT type, COUNT(id)
                    FROM `bigquery-public-data.hacker_news.full`
                    GROUP BY type
                    """
stories_safe = hacker_news.query_to_pandas_safe(stories_query)
print(stories_safe)

#How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)
deleted_query = """
                 SELECT deleted,COUNTIF(deleted != False)
                 FROM `bigquery-public-data.hacker_news.comments`
                 GROUP BY deleted
                 HAVING deleted = True
                 """
deleted_safe = hacker_news.query_to_pandas_safe(deleted_query)
print(deleted_safe)

#Optional extra credit: read about aggregate functions other than COUNT() 
#and modify one of the queries you wrote above to use a different aggregate function.
deleted_query_2 = """
                 SELECT deleted,COUNTIF(deleted != False)
                 FROM `bigquery-public-data.hacker_news.comments`
                 GROUP BY deleted
                 HAVING deleted = True
                 """
deleted_safe_2 = hacker_news.query_to_pandas_safe(deleted_query_2)
print("another method")
print(deleted_safe_2)
