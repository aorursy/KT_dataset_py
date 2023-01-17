#Load BigQuery Helper: https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper
# And Pandas:
import pandas as pd
# Create Big Query Helper Object
bq_day2 = BigQueryHelper("bigquery-public-data", "hacker_news")
#List all Tables of the Dataset
bq_day2.list_tables()
#Lets view the Table Schema
bq_day2.table_schema("full")
# Lets view the first 5 Rows of the full table
bq_day2.head("full", num_rows=5)
QUERY1 = """
        SELECT count(t.id) as unique_stories
        FROM `bigquery-public-data.hacker_news.full` AS t  
        """
# Estimate Query size:
bq_day2.estimate_query_size(QUERY1)
# Excecute the SQL (safely, not more than 1GB) and put the Result set into a pandas Dataframe (df)
df_unique_stories = bq_day2.query_to_pandas_safe(QUERY1)
#lets show the Datframe
df_unique_stories
QUERY2 = """
        SELECT t.deleted, count(t.id) as unique_stories
        FROM `bigquery-public-data.hacker_news.full` AS t 
        WHERE t.deleted = True
        GROUP BY t.deleted
        """
# Estimate Query #2 size:
bq_day2.estimate_query_size(QUERY2)
# Excecute the Query #2 (safely, not more than 1GB) and put the Result set into a pandas Dataframe (df2)
df = bq_day2.query_to_pandas_safe(QUERY2)
#lets show the Datframe
df