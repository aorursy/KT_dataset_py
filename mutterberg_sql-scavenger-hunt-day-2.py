import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")

# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("full")
# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
hacker_news.estimate_query_size(query)
# only run this query if it's less than 300 MB
stories_by_type = hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.3)
stories_by_type.to_csv("stories_by_type.csv")
# query to pass to 
query2 = """SELECT COUNT(*)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            GROUP BY deleted
        """
hacker_news.estimate_query_size(query2)
# only run this query if it's less than 100 MB
del_bool_vals = hacker_news.query_to_pandas_safe(query2, max_gb_scanned=0.1)
del_bool_vals.to_csv("del_bool_vals.csv")