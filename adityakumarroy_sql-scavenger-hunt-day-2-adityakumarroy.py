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
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
#Stories for each type
query_stories_each_type = """SELECT type, count(*)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories_each_type = hacker_news.query_to_pandas_safe(query_stories_each_type)
print (stories_each_type)

#Number of comments deleted
query_comments_deleted= """SELECT count(*) 
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted IS NOT True
        """
comments_deleted = hacker_news.query_to_pandas_safe(query_comments_deleted)
print (comments_deleted)