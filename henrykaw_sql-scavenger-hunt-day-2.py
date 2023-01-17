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
#Print the list of tables in hacker news
hacker_news.list_tables()
#print the first few rows
hacker_news.head('full')
#Query to pass
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
#estimate size of query
hacker_news.estimate_query_size(query1)
#Query to pandas dataframe
story_types = hacker_news.query_to_pandas_safe(query1)
story_types.shape
story_types.rename(index=str, columns={"type": "type", "f0_": "amount"}, inplace=True)
story_types
#Query to pass
query2 = """SELECT deleted, COUNT(id) AS amount_deleted
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
            """
#estimate size of query
hacker_news.estimate_query_size(query2)
#get pandas dataframe for count of deleted comments
comments_deleted = hacker_news.query_to_pandas_safe(query2)

#print count
comments_deleted
#Query to pass
query3 = """SELECT COUNTIF( deleted = True)
            FROM `bigquery-public-data.hacker_news.comments`
            """
#estimate query size
hacker_news.estimate_query_size(query3)
#make a dataframe and print it
alt_deleted_comments = hacker_news.query_to_pandas_safe(query3)
alt_deleted_comments