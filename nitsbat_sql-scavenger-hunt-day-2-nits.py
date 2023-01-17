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
hacker_news.head('full')

hacker_news.table_schema('full')
query2 = """ select type,count(id) from `bigquery-public-data.hacker_news.full`
             group by type """
hacker_news.estimate_query_size(query2)
result = hacker_news.query_to_pandas_safe(query2)
result.head()
hacker_news.table_schema('comments')
hacker_news.head('comments')

query3 = """ select count(deleted) as bro from `bigquery-public-data.hacker_news.comments`
             where deleted=True """
hacker_news.estimate_query_size(query3)
hacker_news.query_to_pandas_safe(query3)
query2 = """ select t1.author,max(t1.total) as Max from 
            (select author as author,count(author) as total from
            `bigquery-public-data.hacker_news.comments`
             group by author ) t1
             group by t1.author
             having max(t1.total) = (select max(t2.total) as total from
                                    (
                                        select author as author,count(author) as total from
                                        `bigquery-public-data.hacker_news.comments`
                                         group by author
                                    ) t2    
                                    ) """
hacker_news.estimate_query_size(query2)
hacker_news.query_to_pandas_safe(query2)
