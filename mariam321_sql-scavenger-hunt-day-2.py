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
#import the bigdataquery package
import bq_helper
hacker_news_helper=bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
query="""SELECT COUNT (id),type 
         from `bigquery-public-data.hacker_news.full`
         group by type"""
        
type_count=hacker_news_helper.query_to_pandas_safe(query) 
print(type_count)
#count of deleted comments
query="""SELECT count(deleted) from `bigquery-public-data.hacker_news.comments`  
       group by deleted having deleted is true """
deleted_comments_count=hacker_news_helper.query_to_pandas_safe(query)
print(deleted_comments_count)
