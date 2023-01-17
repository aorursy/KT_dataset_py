# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset (OpenAQ = openaq)
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
# print information on all the columns in the "full" table
# in the hackernews dataset
hacker_news.table_schema("full")
# preview the first couple lines of the "full" table (using head())
hacker_news.head("full")
# first query: number news of each type
query1 = """SELECT type, COUNT(id) as count_id
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type """
# check how big this query will be (using estimate_query_size())
hacker_news.estimate_query_size(query1)
# check out the number of unique stories (if the query is smaller than 1 gig)
no_unique_stories = hacker_news.query_to_pandas_safe(query1)
# save our dataframe as a .csv 
no_unique_stories.to_csv("no_unique_stories.csv")
# second query: How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query2 = """SELECT COUNT(id) as number_deleted_comment
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "comment" AND deleted = True """
# check how big this query will be (using estimate_query_size())
hacker_news.estimate_query_size(query2)
# check out number of deleted comment (if the query is smaller than 1 gig)
no_deleted_comment = hacker_news.query_to_pandas_safe(query2)
# save our dataframe as a .csv 
no_deleted_comment.to_csv("no_deleted_comment.csv")