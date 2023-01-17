# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head('comments')

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
hacker_news.table_schema("full")
# Your code goes here :)

query1 = """
        select count(distinct id), type
        from `bigquery-public-data.hacker_news.full`
        group by type
        """
# run query
stories_cnt = hacker_news.query_to_pandas_safe(query1)

stories_cnt.head()
hacker_news.estimate_query_size(query1)
stories_cnt.to_csv('stories_cnt.csv')

hacker_news.table_schema("comments")
query2 = """
        select count(distinct id), deleted
        from `bigquery-public-data.hacker_news.comments`
        group by deleted 
        having deleted = True
        """
# run query
deleted_comments_cnt = hacker_news.query_to_pandas_safe(query2)
deleted_comments_cnt.head()

query3 = """
        select countif(deleted = True), deleted
        from `bigquery-public-data.hacker_news.comments`
        group by deleted
        """
# run query
deleted_comments_cnt1 = hacker_news.query_to_pandas_safe(query3)
deleted_comments_cnt1.head()
query4 = """
        select sum(distinct id), type
        from `bigquery-public-data.hacker_news.full`
        group by type
        """
stories_cnt1 = hacker_news.query_to_pandas_safe(query1)

stories_cnt1.head()
