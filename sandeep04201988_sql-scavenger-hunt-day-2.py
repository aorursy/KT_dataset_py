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
hacker_news.head('full')
query = """select type,count(id) as total from 
            `bigquery-public-data.hacker_news.full`
            group by type
        """
hacker_news.estimate_query_size(query)
q1 = hacker_news.query_to_pandas_safe(query)
q1
hacker_news.head('comments')
query = """select deleted,count(id) as Deleted_Count from 
            `bigquery-public-data.hacker_news.comments`
            #where deleted = True
            group by deleted
        """
hacker_news.estimate_query_size(query)
q2 = hacker_news.query_to_pandas_safe(query)
q2
q2.loc[q2['deleted'] == True]['Deleted_Count']
query = """select min(id) as oldest_comment_to_be_deleted from 
            `bigquery-public-data.hacker_news.comments`
            where deleted = True
        """
q3 = hacker_news.query_to_pandas_safe(query)
q3
query = """select max(id) as latest_comment_to_be_deleted from 
            `bigquery-public-data.hacker_news.comments`
            where deleted = True
        """
q3_1 = hacker_news.query_to_pandas_safe(query)
q3_1