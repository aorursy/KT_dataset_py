# import our bq_helper package
import bq_helper 

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# print all the tables in this dataset (we need 'full')
hacker_news.list_tables()

# print the first couple rows of the "full" table
hacker_news.head("full")

# print information on all the columns in the "full" table
# in the open_aq dataset
hacker_news.table_schema("full")

# How many unique stories (use the “id” column) are there in the full table
query_unique_id = """
SELECT COUNT(id) AS unique_stories
FROM `bigquery-public-data.hacker_news.full`
"""
# How many comments have been deleted? 
query_count_del_comment = """
SELECT COUNT(id) as delete_comment
FROM `bigquery-public-data.hacker_news.comments`
WHERE deleted = TRUE
"""

# check how big this query will be
hacker_news.estimate_query_size(query_unique_id)

# only run this query if it's less than 1 gigabyte by default
job_1 = hacker_news.query_to_pandas_safe(query_unique_id)

job_1.head()

# save our dataframe as a .csv 
job_1.to_csv("SQL_Scavenger_Hunt_task3.csv")

# check how big this query will be
hacker_news.estimate_query_size(query_count_del_comment)

# only run this query if it's less than 1 gigabyte by default
job_2 = hacker_news.query_to_pandas_safe(query_count_del_comment)

job_2.head()

# save our dataframe as a .csv 
job_2.to_csv("SQL_Scavenger_Hunt_task4.csv")