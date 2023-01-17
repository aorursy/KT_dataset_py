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

# query to pass to, How many stories (use the "id" column) are there of each type (in the "type" column) in the full table? 

#view the full table of comments
hacker_news.head("full")

query = """SELECT type, COUNT(id) AS total_by_type
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
    ORDER BY total_by_type DESC
        """

number_of_stories = hacker_news.query_to_pandas_safe(query)

number_of_stories.head()
query = """SELECT COUNT(ID) as number_of_deleted_comments
    FROM `bigquery-public-data.hacker_news.comments`
    GROUP BY deleted
    Having deleted = True
        """

number_of_deleted_comments = hacker_news.query_to_pandas_safe(query)

number_of_deleted_comments.head()