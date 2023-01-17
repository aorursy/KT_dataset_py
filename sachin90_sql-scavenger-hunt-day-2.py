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
# Writing Query for count of each type of story
query1= """ SELECT type,COUNT(id) AS story_count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY story_count """

# Getting the result in to the dataframe with pandas safe
story_count_df=hacker_news.query_to_pandas_safe(query1)
# Printing the required result
story_count_df
# Writing the query to get count of deleted comments 
query2= """ SELECT deleted,COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted """
# Saving the query result in dataframe after running in safe mode
del_comments=hacker_news.query_to_pandas_safe(query2)
# Print the result
del_comments
# Query to get the maximum ranking of each author
query3= """ SELECT author,max(ranking) AS max_rank
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author """

# Saving the dataframe into dataframe after running in safe mode
max_ranking_comments=hacker_news.query_to_pandas_safe(query3)
# Printing the top 10 rows from the result
max_ranking_comments.head(10)