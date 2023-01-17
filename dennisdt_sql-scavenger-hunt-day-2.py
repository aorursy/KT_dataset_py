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
# count the "id" column and group by type to obtain number of stories in each type
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# perform the query
type_id_count = hacker_news.query_to_pandas_safe(query)
# display top 5 types and its count
type_id_count.head()
# count all values in deleted column of comments table for values of deleted = True
query = """SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
num_deleted = hacker_news.query_to_pandas_safe(query)
# display total number of deleted comments in the comments table
print(num_deleted)