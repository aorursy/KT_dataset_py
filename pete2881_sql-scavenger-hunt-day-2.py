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
# tables
# what tables are available in the hacker news dataset?
hacker_news.list_tables()


# want to use the "full" table
# list the first few rows (safely)
hacker_news.head("full")
story_query = """SELECT type,count(id) as stories
        FROM `bigquery-public-data.hacker_news.full` 
        GROUP BY type
        ORDER BY stories desc;"""
hacker_news.estimate_query_size(story_query)

# run the query, look at the results
story_df = hacker_news.query_to_pandas(story_query)
story_df.shape
story_df
del_query = """ SELECT count(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "comment" AND deleted=True
            """
# check query size
hacker_news.estimate_query_size(del_query)
deleted_comments = hacker_news.query_to_pandas(del_query)
deleted_comments
first_del_query = """ SELECT min(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "comment" AND deleted=True
            """
del_comment = hacker_news.query_to_pandas(first_del_query)
del_comment