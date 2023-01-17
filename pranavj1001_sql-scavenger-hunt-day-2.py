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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")


# print all the tables in this dataset
# hacker_news.list_tables()

# print the first couple rows of the "full" table
hacker_news.head("full")

# query for the first question
query_for_first_question = """SELECT type, COUNT(id)
                              FROM `bigquery-public-data.hacker_news.full`
                              GROUP BY type
                           """

# check the size of the query
# hacker_news.estimate_query_size(query_for_first_question)

# run the query
stories_number = hacker_news.query_to_pandas_safe(query_for_first_question)
# print number of stories
stories_number

# query for the second question
query_for_second_question = """SELECT deleted, COUNT(id)
                               FROM `bigquery-public-data.hacker_news.full`
                               GROUP BY deleted
                               HAVING deleted = True
                            """

# check the size of the query
# hacker_news.estimate_query_size(query_for_second_question)

# run the query
deleted_comments_number = hacker_news.query_to_pandas_safe(query_for_second_question)
# print number of deleted_comments
deleted_comments_number

# optional query
optional_query = """SELECT type, COUNTIF(score > 3)
                    FROM `bigquery-public-data.hacker_news.full`
                    GROUP BY type
                 """

# check the size of the query
hacker_news.estimate_query_size(optional_query)

# run the query
optional = hacker_news.query_to_pandas_safe(optional_query)
# print number of stories which have a score of greater than 3
optional