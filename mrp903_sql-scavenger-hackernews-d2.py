import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
hackernews = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', 
                                      dataset_name = 'hacker_news')
hackernews.list_tables()
hackernews.table_schema(table_name='full')
## Sample query:
query_0 = """SELECT DISTINCT(type)
             FROM `bigquery-public-data.hacker_news.full`
           """
hackernews.estimate_query_size(query_0)
sample_query = hackernews.query_to_pandas(query = query_0)
sample_query.head(20)
query_1 = """ SELECT COUNT(DISTINCT(id)) AS DistinctStories
              FROM `bigquery-public-data.hacker_news.full`
              WHERE type = 'story'
            """
print ('Size of the query is:')
hackernews.estimate_query_size(query_1)
unique_stories = hackernews.query_to_pandas(query = query_1)
print ('Following shows the #unique stories:')
unique_stories
query_2 = """SELECT COUNT(deleted) as Deleted
             FROM `bigquery-public-data.hacker_news.full`
             WHERE deleted IS True
          """
hackernews.estimate_query_size(query_2)
deleted_stories = hackernews.query_to_pandas(query_2)
print ('#deleted stories is:')
deleted_stories
