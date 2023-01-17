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
stories_query = """Select type, count(*) No_of_stories
                    from `bigquery-public-data.hacker_news.full`
                    group by type
                """
stories_data = hacker_news.query_to_pandas_safe(stories_query)
stories_data
# Simple way of calculating number of deleted comments by using counif()
comm_query = """Select countif(deleted=True) No_deleted_comms
                from `bigquery-public-data.hacker_news.comments`
            """
deleted_comm = hacker_news.query_to_pandas_safe(comm_query)
# Number of delted comments is:
print(deleted_comm)
# Now, I am going to use if() function in combination with sum()
# so that if deleted = True -> value is 1, otherwise 0
comments_query = """Select sum(if(deleted = True, 1, 0)) No_comments_deleted
                    from `bigquery-public-data.hacker_news.comments`
                """
comments_data = hacker_news.query_to_pandas_safe(comments_query)
# We should have same value as before:
print(comments_data)