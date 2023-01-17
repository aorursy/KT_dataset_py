# borrowed from Rachael Tatman 
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
query = """ SELECT type, count(id) as cnt_story_id
            FROM `bigquery-public-data.hacker_news.full`
            GROUP by 1
        """ 

stories_by_type = hacker_news.query_to_pandas_safe(query)

stories_by_type
query = """ SELECT deleted, count(id) as cnt_comments
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted is True
            GROUP BY 1
        """ 

comments = hacker_news.query_to_pandas_safe(query)

comments
query = """ SELECT type, sum(1) as cnt_story_records
            FROM `bigquery-public-data.hacker_news.full`
            GROUP by 1
        
        """ 

story_recs = hacker_news.query_to_pandas_safe(query)

story_recs