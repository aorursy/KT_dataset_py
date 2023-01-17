import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",dataset_name = "hacker_news")
hacker_news.list_tables()
hacker_news.table_schema("full")
hacker_news.head("full")
hacker_news.head("full",selected_columns="by",num_rows = 10)
query = """ SELECT score from `bigquery-public-data.hacker_news.full` where type = "job" """
hacker_news.estimate_query_size(query)
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
list = hacker_news.query_to_pandas_safe(query)
list.score.mean()
list.to_csv("FirstBigQuery.csv")
