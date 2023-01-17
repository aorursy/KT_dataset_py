import bq_helper 
# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
# print information on all the columns in the "comments" table
# in the hacker_news dataset
hacker_news.table_schema("comments")
# preview the first couple lines of the "comments" table
hacker_news.head("comments")
# preview the first ten entries in the 'text' column of the 'comments' table
hacker_news.head("comments", selected_columns="text", num_rows=10)
# BigQueryHelper.estimate_query_size() - to estimate how big your query will be 
# before you actually execute it. 

# This query looks in the 'comments' table in the hacker_news
# dataset, then gets the 'text' column from every row where 
# the dead column has "None" in it.
query = """SELECT text
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE dead != "None" """

# check how big this query will be
#hacker_news.estimate_query_size(query)


# This query looks in the 'comments' table in the hacker_news
# dataset, then gets the 'by' column from every row where 
# the 'author' column has "lv" in it.

# 'by' is in quotes because 'by' is SQL specific function similar to 'from', 'where' etc
query1 = """SELECT author
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE 'by' = "lv" """
# check how big this query will be
hacker_news.estimate_query_size(query1)
# this query looks in the full table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query2 = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
hacker_news.estimate_query_size(query2)
query3 = """SELECT author
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE ranking != 0 """
hacker_news.estimate_query_size(query3)
# only run this query if it's less than 100 MB
hacker_news.query_to_pandas_safe(query3, max_gb_scanned=0.1)
# check out the scores of author ranking (if the 
# query is smaller than 1 gig)
author_rankings = hacker_news.query_to_pandas_safe(query3)
# average score for author_rankings
author_rankings.score.mean()