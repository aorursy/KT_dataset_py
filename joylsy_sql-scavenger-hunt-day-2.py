# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")
# query to pass to get how many unique stories (use the “id” column) are there in the full table
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            where type = 'story'
            GROUP BY type  
        """

full_unique_stories = hacker_news.query_to_pandas_safe(query)

full_unique_stories.head()
# query to pass to get comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """SELECT type,COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            where deleted = True
            group by type 
            having type = 'comment'     
        """

full_deleted_comments = hacker_news.query_to_pandas_safe(query)

full_deleted_comments.head()
query = """SELECT count(id)
            FROM `bigquery-public-data.hacker_news.comments`
            where deleted = True
        """

comments_deleted = hacker_news.query_to_pandas_safe(query)

comments_deleted.head()
query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            where type = 'comment'
        """

full_deleted_comments = hacker_news.query_to_pandas_safe(query)

full_deleted_comments.head()
# query to pass to get comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """ SELECT type,COUNTIF(deleted is null), COUNTIF(deleted = True), COUNTIF(deleted = False)
            FROM `bigquery-public-data.hacker_news.full`
            group by type 
            having type = 'comment'     
        """

full_deleted_comments = hacker_news.query_to_pandas_safe(query)

full_deleted_comments.head()
# query to pass to get comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """ SELECT COUNTIF(deleted is null), COUNTIF(deleted = True), COUNTIF(deleted = False)
            FROM `bigquery-public-data.hacker_news.full`
            where type = 'comment'    
        """

full_deleted_comments = hacker_news.query_to_pandas_safe(query)

full_deleted_comments.head()
query = """SELECT COUNTIF(deleted is null), COUNTIF(deleted = True), COUNTIF(deleted = False)
            FROM `bigquery-public-data.hacker_news.comments`
        """

comments_deleted = hacker_news.query_to_pandas_safe(query)
comments_deleted.head()