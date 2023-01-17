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
hacker_news.list_tables()
hacker_news.head("full")

queryCountIdsByType = """SELECT TYPE, COUNT(ID) CountOfStories
                        FROM `bigquery-public-data.hacker_news.full`
                        GROUP BY TYPE
"""

countStoriesByType = hacker_news.query_to_pandas_safe(queryCountIdsByType)
countStoriesByType.head()
hacker_news.table_schema("comments")
queryNumOfDeletedComment = """SELECT count(*) NumOfDeletedComment
                        FROM `bigquery-public-data.hacker_news.comments`
                        WHERE deleted 
"""
NumOfDeletedComment = hacker_news.query_to_pandas(queryNumOfDeletedComment)
NumOfDeletedComment
queryAvgScoreByType = """SELECT TYPE, AVG(SCORE) AvgScoreByType
                        FROM `bigquery-public-data.hacker_news.full`
                        
                        GROUP BY TYPE
                        ORDER BY AvgScoreByType DESC
"""
AvgScoreByType = hacker_news.query_to_pandas_safe(queryAvgScoreByType)
AvgScoreByType