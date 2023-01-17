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
import bq_helper

query = """
        SELECT type, COUNT(type) AS count
        FROM `bigquery-public-data.hacker_news.full_201510`
        GROUP BY type
        """
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
types = hacker_news.query_to_pandas_safe(query)

query2 = """
        SELECT type, COUNT(deleted) AS deleted
        FROM `bigquery-public-data.hacker_news.full_201510`
        GROUP BY type
        HAVING type = 'comment'
        """
hacker_news2 = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
deleted = hacker_news2.query_to_pandas_safe(query2)
print(types)
print('---------------------------')
print(deleted)