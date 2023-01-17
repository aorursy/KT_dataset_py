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
#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query_one =  """SELECT type, COUNT(id)
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
             """
type_count = hacker_news.query_to_pandas_safe(query_one)
type_count.head()
#How many comments have been deleted?
#without using 'HAVING'
query_two =  """SELECT deleted, COUNT(id)
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY deleted
             """
number_of_deleted_comments = hacker_news.query_to_pandas_safe(query_two)
number_of_deleted_comments

query_three =  """SELECT deleted, COUNT(ID)
                  FROM `bigquery-public-data.hacker_news.comments`
                  GROUP BY deleted
                  HAVING deleted = TRUE
               """
number_of_deleted_comments_two = hacker_news.query_to_pandas_safe(query_three)
number_of_deleted_comments_two

#or you could use WHERE before the aggregation
query_four =  """SELECT deleted, COUNT(ID)
                 FROM `bigquery-public-data.hacker_news.comments`
                 WHERE deleted = TRUE
                 GROUP BY deleted
              """
number_of_deleted_comments_three = hacker_news.query_to_pandas_safe(query_four)
number_of_deleted_comments_three


##
query_five =  """SELECT type, COUNT(id)
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                ORDER BY COUNT(id) DESC
             """
type_ordered = hacker_news.query_to_pandas_safe(query_five)
type_ordered.head()