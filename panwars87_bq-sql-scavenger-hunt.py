# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
hacker_news.table_schema("full")
hacker_news.head("comments")
hacker_news.head("full", selected_columns="by", num_rows=10)
query = """ select title, score from `bigquery-public-data.hacker_news.full` where type="job" """
hacker_news.estimate_query_size(query)
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
# check out the scores of job postings (if the 
# query is smaller than 1 gig)
job_post_scores = hacker_news.query_to_pandas_safe(query)
job_post_scores.score.max()
job_post_scores.score > 0
