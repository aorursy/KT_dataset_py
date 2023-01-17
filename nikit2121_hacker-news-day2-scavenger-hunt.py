import pandas as pd
from bq_helper import BigQueryHelper
bq_hacker_news = BigQueryHelper(active_project="bigquery-public-data",dataset_name="hacker_news")
bq_hacker_news.list_tables()
bq_hacker_news.head('comments',num_rows=10)
query = """select parent,count(id) as total_replies
            from `bigquery-public-data.hacker_news.comments`
            group by parent
            having count(id)>10
        """
bq_hacker_news.estimate_query_size(query)
popular_stories = bq_hacker_news.query_to_pandas_safe(query)
popular_stories.head()
bq_hacker_news.head('full',num_rows=50)
query = """select type,count(id) as total_stories
        from `bigquery-public-data.hacker_news.full`
        group by type
        """
bq_hacker_news.estimate_query_size(query)
stories = bq_hacker_news.query_to_pandas(query)
stories.head()
bq_hacker_news.table_schema('comments')
query = """select count(id) as total_deleted_comments
            from `bigquery-public-data.hacker_news.comments`
            where deleted=TRUE

        """
bq_hacker_news.estimate_query_size(query)
deleted = bq_hacker_news.query_to_pandas(query)
deleted.head()
