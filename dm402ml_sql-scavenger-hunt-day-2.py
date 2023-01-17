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
# Daniela's code goes here :)
#Print list of tables
hacker_news.list_tables()
#Get name of columns for table "full"
hacker_news.head("full")

#1-How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query_1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
#Safe method
stories_per_type = hacker_news.query_to_pandas_safe(query_1)

print(stories_per_type)
#2-How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments 
#table will have the value "True".)
#Get name of columns for table "comments"
hacker_news.head("comments")

query_2 = """SELECT deleted, COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted=True
        """
#Safe method
comments_deleted = hacker_news.query_to_pandas_safe(query_2)

print(comments_deleted)
#3-Optional extra credit: AVG() aggregate function to calculate the average score per each type
query_3 = """SELECT type, AVG(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
#Safe method
scores_per_type = hacker_news.query_to_pandas_safe(query_3)

print(scores_per_type)