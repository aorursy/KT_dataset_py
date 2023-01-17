# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                       dataset_name="hacker_news")
# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
# print information on all the columns in a table of the dataset
tbl = 'comments'
hacker_news.table_schema(tbl)
# print the first couple rows of the "comments" table
hacker_news.head(tbl)
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
# D2Q1
# query to pass to 
query1 = """SELECT type, COUNT(id) 
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

full_by_type = hacker_news.query_to_pandas_safe(query1)
full_by_type.shape
full_by_type
# D2Q2
query2 = """SELECT deleted, COUNT(deleted) AS tot
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments
# D2_extra credit: modify one of the queries you wrote above to use a different aggregate function.
query3 = """SELECT parent, COUNT(id), AVG(ranking) 
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent 
            HAVING COUNT(id) > 20
        """
aver_ranking_by_parent = hacker_news.query_to_pandas_safe(query3)
aver_ranking_by_parent.shape