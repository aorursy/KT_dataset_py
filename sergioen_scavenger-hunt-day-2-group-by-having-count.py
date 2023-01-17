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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# Your code goes here :)
# Question 1
#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

qry="""SELECT type, COUNT(id) AS count
       FROM `bigquery-public-data.hacker_news.full`
       GROUP BY type"""
# Question 1
stories_for_type = hacker_news.query_to_pandas_safe(qry)
# Question 1
stories_for_type.head()
##How many comments have been deleted?
# Question 2
qry="""SELECT type, COUNT(id) AS deleted
         FROM `bigquery-public-data.hacker_news.full`
        WHERE deleted IS TRUE
       GROUP BY type"""
# Question 2
stories_for_type = hacker_news.query_to_pandas_safe(qry)
# Question 2
stories_for_type.head()
qry="""SELECT type,
              sum(valid) AS valid,
              sum(deleted) AS deleted,
              (sum(valid)+sum(deleted)) AS total
         FROM (SELECT type, COUNT(id) AS valid, 0 AS deleted
                 FROM `bigquery-public-data.hacker_news.full`
                WHERE deleted IS NOT TRUE
                GROUP BY 1
               UNION ALL
               SELECT type, 0 AS valid , COUNT(id) AS deleted
                 FROM `bigquery-public-data.hacker_news.full`
                WHERE deleted IS TRUE
                GROUP BY 1)
         GROUP BY 1"""
stories_for_type = hacker_news.query_to_pandas_safe(qry)
stories_for_type