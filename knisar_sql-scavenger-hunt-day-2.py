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

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.head("full")
query = """SELECT type, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories_count = hacker_news.query_to_pandas_safe(query)

query1 = """SELECT COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
        """
comments_deleted = hacker_news.query_to_pandas_safe(query1)

print("The different types of stories with their count are:\n",stories_count)
print("The number of comments deleted are :",comments_deleted['count'][0])
