








#import package with helper function
import bq_helper

#creating a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="hacker_news")

#printing first couple of rows of full table
hacker_news.head("full")


#query to see how many stories are there for each type
query = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type"""

#query will execute only if its less than 1GB
story_type_count = hacker_news.query_to_pandas_safe(query)

#display the results
story_type_count.head
#query to see how many comments have been deleted

#If a comment is deleted - it will have "True" value in the "deleted"' column in the "comments" table
#Taking a look at the comments table
hacker_news.head("comments")
query1 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True """

#query will execute only if its less than 1GB
deleted_comments = hacker_news.query_to_pandas_safe(query1)

#display the results
deleted_comments.head


