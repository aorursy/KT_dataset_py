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
# Question 1
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

# I don't see the column 'type' in the data header above, so I'm changing this to 'author'
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")


my_query = """SELECT type, COUNT(id) AS ID_Count
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                ORDER BY ID_Count DESC
        """

# OR

my_query = """SELECT type, AVG(id) AS ID_Avg
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                ORDER BY ID_Avg DESC
        """

ID_table = hacker_news.query_to_pandas_safe(my_query)

ID_table
# How many comments have been deleted? (If a comment was deleted the "deleted" column 
# in the comments table will have the value "True"

deleted_query = """SELECT deleted, COUNT(id)
                    FROM `bigquery-public-data.hacker_news.comments`
                    GROUP BY deleted
                      """

#OR

deleted_query = """SELECT deleted,COUNT(id) AS ID_Count
                    FROM `bigquery-public-data.hacker_news.comments`
                    GROUP BY deleted
                    HAVING deleted=True
                      """


deleted_table = hacker_news.query_to_pandas_safe(deleted_query)

deleted_table