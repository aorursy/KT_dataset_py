import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project= " bigquery-public-data", dataset_name = " hacker_news")
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
hacker_news.list_tables()
#print information ona ll the column in the "full" table
#in the hacker_news dataset
hacker_news.table_schema("full")