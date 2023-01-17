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

# check again what columns the full table consists of
hacker_news.head("full")
# the query 

query2 = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
        """
# saving the result of the query to a Pandas dataset
stories_by_type = hacker_news.query_to_pandas_safe(query2)

# displaying the result
print(stories_by_type)
# taking a look at the comments table
hacker_news.head("comments")
# the query
query3 = """SELECT deleted, COUNT(id)
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY deleted
           HAVING deleted = TRUE
        """
# saving the query as a Pandas dataframe
deleted_comments = hacker_news.query_to_pandas_safe(query3)

# returning the result
print(deleted_comments)
# we use the full table again for the query
query4 = """SELECT type, MAX(score)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
           ORDER BY MAX(score)
        """
# saving the result of the query in a Pandas dataframe
highest_score = hacker_news.query_to_pandas_safe(query4)

# return the highest scores per content type
print(highest_score)