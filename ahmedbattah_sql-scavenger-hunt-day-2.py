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
# Number of stories by type
stories_query = """ SELECT type, COUNT(id)
                    FROM `bigquery-public-data.hacker_news.full`
                    GROUP BY type """
stories_df = hacker_news.query_to_pandas_safe(stories_query)
stories_df.columns = ['type', 'count']
stories_df.head()
# Deleted comments
del_comments_query = """ SELECT COUNT(id)
                         FROM `bigquery-public-data.hacker_news.comments`
                         WHERE deleted = True """
del_comments_df = hacker_news.query_to_pandas_safe(del_comments_query)
del_comments_df.columns = ['No. of deleted comm.']
del_comments_df.head()
avg_score_query = """ SELECT type, AVG(score)
                      FROM `bigquery-public-data.hacker_news.full`
                      GROUP BY type
                      """
avg_score_df = hacker_news.query_to_pandas_safe(avg_score_query)
avg_score_df.columns = ['type', 'avg_score']
avg_score_df.head()