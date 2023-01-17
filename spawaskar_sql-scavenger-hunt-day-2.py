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

hacker_news.head("full")

#Query to get number of unique stories
queryStories = """SELECT count(DISTINCT id) AS noOfType
                   FROM `bigquery-public-data.hacker_news.full` 
                   WHERE UPPER(type) = 'STORY'
                """
noOf_stories = hacker_news.query_to_pandas_safe(queryStories)
noOf_stories.head()
#No of comments that have been deleted

#hacker_news.head("comments")

queryDelComments = """SELECT count(deleted) AS numDel
                       FROM `bigquery-public-data.hacker_news.comments` 
                       WHERE deleted = True
                """
noOfDelComm = hacker_news.query_to_pandas_safe(queryDelComments)
noOfDelComm.head()
#Extra credit
query = """SELECT parent, COUNT(id) AS count, MIN(ID) AS minID, MAX(id) AS maxID
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10 
        """
popular_stories_avg = hacker_news.query_to_pandas_safe(query)
popular_stories_avg.head()