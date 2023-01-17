# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments",10)
# Your Code Here
query = ''' select type,count(id) from `bigquery-public-data.hacker_news.full` group by type '''

stories = hacker_news.query_to_pandas_safe(query)

stories
# Your Code Here
query = '''select count(id) from `bigquery-public-data.hacker_news.full` where deleted = True '''
delet = hacker_news.query_to_pandas_safe(query)
delet
# Your Code Here
query = ''' select avg(score) from `bigquery-public-data.hacker_news.full` where score >= 0 '''
aggre = hacker_news.query_to_pandas_safe(query)
aggre