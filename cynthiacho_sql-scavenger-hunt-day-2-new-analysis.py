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
#query to count unique stories of each type
"""SELECT type, COUNT DISTINCT(id) as Count_Stories
   FROM bigquery-public-data.hacker_news.full
   GROUP BY Stories"""



#query to count average number of stories of each type
"""SELECT type, AVG(id) as Average_Stories
   FROM bigquery-public-data.hacker_news.full
   GROUP BY Average_Stories"""

#query to identify number of deleted comments
"""SELECT comments, COUNT(deleted) AS Number of deleted comments
   FROM bigquery-public-data.hacker_news.comments
   WHERE deleted="True"
"""
#query to identify total number of deleted comments
"""SELECT comments, AVG(deleted) AS Average Number of deleted comments
   FROM bigquery-public-data.hacker_news.comments
   WHERE deleted = "True"
"""