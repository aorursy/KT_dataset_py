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
# How many stories (use the "id" column) are there
# of each type (in the "type" column)in the full table?

type_query = """
                SELECT type, COUNT(id) AS stories_count
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                """
story_types = hacker_news.query_to_pandas_safe(type_query)

print(story_types)

# How many comments have been deleted?
# (If a comment was deleted the "deleted" column
# in the comments table will have the value "True".)

deleted_query = """
                SELECT deleted, COUNT(id) AS comment_count
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY deleted
                HAVING deleted = True
                """
deleted = hacker_news.query_to_pandas_safe(deleted_query)

print(deleted)

# modify one of the queries you wrote above to use a different aggregate function
# What is the average number of comments on a parent post?


avg_query = """ SELECT AVG(comment_count) as average_comments
                FROM
                (
                    SELECT parent, COUNT(id) as comment_count
                    FROM `bigquery-public-data.hacker_news.comments`
                    GROUP BY parent
                )
            """
average = hacker_news.query_to_pandas_safe(avg_query)

print(average)