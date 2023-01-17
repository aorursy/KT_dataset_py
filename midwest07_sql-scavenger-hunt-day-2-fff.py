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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="hacker_news")

hacker_news.list_tables()

hacker_news.head("full",5)


query1= """ SELECT type, COUNT(id)
             FROM  `bigquery-public-data.hacker_news.full`
             GROUP BY type
        """


hacker_news.estimate_query_size(query1)
full_type=hacker_news.query_to_pandas_safe(query1)


full_type.to_csv("full_type")
hacker_news.head("full",10)
query2= """ SELECT  deleted, COUNT(deleted)
             FROM  `bigquery-public-data.hacker_news.full`
             GROUP BY deleted
             having deleted is TRUE
            
        """
hacker_news.estimate_query_size(query2)
deleted_true=hacker_news.query_to_pandas_safe(query2)
deleted_true.to_csv("deleted_true")
query3= """ SELECT deleted, COUNTIF(deleted is TRUE)
            FROM `bigquery-public-data.hacker_news.full`
            group by deleted
        """
hacker_news.estimate_query_size(query3)
countif_deleted_true=hacker_news.query_to_pandas_safe(query3)
countif_deleted_true.to_csv("countif_deleted_true")