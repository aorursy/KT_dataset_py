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
hacker_news.list_tables()
hacker_news.head("full")
# How many stories are there by each type in the type column 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
types_count = hacker_news.query_to_pandas_safe(query)

%pylab inline 
print(types_count)
types_count.plot.bar(x='type',y='f0_')
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted 
        """
deleted_count = hacker_news.query_to_pandas_safe(query)
print('The number of deleted posts is: %d'%deleted_count.ix[0].f0_)

query = """
            SELECT type, AVG(descendants)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_rank = hacker_news.query_to_pandas_safe(query)
print(type_rank)