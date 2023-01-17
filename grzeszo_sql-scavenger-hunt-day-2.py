#we'll need that
import bq_helper
# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")

#all the tables in the hacker_news dataset, also check the connection
hacker_news.list_tables()
query1 = """SELECT COUNT(DISTINCT id) AS UniqueIDs
            FROM `bigquery-public-data.hacker_news.full`
            """
unique_ids = hacker_news.query_to_pandas_safe(query1)
#answer 1 - number of unique ids
unique_ids
query2 = """SELECT COUNT(*) AS DeletedComments
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            """
deleted = hacker_news.query_to_pandas_safe(query2)
#answer 2 - number of deleted comments
deleted
