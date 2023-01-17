# To start, we first import the package with helper functions 
import bq_helper

# Then, we create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")
# To know how many stories are there of each type we count how many different ID's are for type
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
type_stories = hacker_news.query_to_pandas_safe(query)
type_stories.head()
# To know how many comments have been deleted
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            GROUP BY deleted
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments.head()
# To know the top commenters 
query3 = """SELECT author, COUNT(id)
            FROM `bigquery-public-data.hacker_news.stories`
            GROUP BY author
        """
top_commenters = hacker_news.query_to_pandas_safe(query3)
top_commenters.head()