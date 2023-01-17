import bq_helper 
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data",dataset_name = "hacker_news")
hacker_news.list_tables()
hacker_news.table_schema("stories")
query = """SELECT * FROM `bigquery-public-data.hacker_news.stories` order by id limit 5000 """

# check how big this query will be
hacker_news.estimate_query_size(query)

hacker_news.query_to_pandas_safe(query)
job_post_scores = hacker_news.query_to_pandas_safe(query)
job_post_scores.to_csv("job_post_scores.csv")