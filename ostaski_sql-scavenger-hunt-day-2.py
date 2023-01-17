# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# import the Big Query helper package
import bq_helper

# create the helper object using bigquery-public-data.hacker_news
bqh = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")

# looking at the data
bqh.list_tables()
bqh.table_schema("full")
bqh.head("full")
bqh.table_schema("comments")
bqh.head("comments")

# Q1. How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# estimate query size
bqh.estimate_query_size(query)

# run a "safe" query and store the resultset into a dataframe
count_by_type = bqh.query_to_pandas_safe(query)

# taking a look
print(count_by_type)
# 13487115 comments, 2845239 stories, 11806 pollopts, 10159 jobs and 1728 polls
# heh, looks like the numbers have changed a bit from yesterday's full table

# saving this in case we need it later
count_by_type.to_csv("count_by_type.csv")

# Q2. How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = TRUE
        """

# estimate query size
bqh.estimate_query_size(query)

# run a "safe" query and store the resultset into a dataframe
count_deleted_comments = bqh.query_to_pandas_safe(query)

# taking a look
print(count_deleted_comments)
# 227736 deleted comments

# saving this in case we need it later
count_deleted_comments.to_csv("count_deleted_comments.csv")

# Optional Q. Modify one of the queries you wrote above to use a different aggregate function.
# let's find the average score of "stories" 
query = """SELECT type, AVG(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING type = "story"
        """

# estimate query size
bqh.estimate_query_size(query)

# run a "safe" query and store the resultset into a dataframe
stories_avg_score = bqh.query_to_pandas_safe(query)

# taking a look
print(stories_avg_score)
# average score for all "stories" is 11.139652
# this value has increased slightly from yesterday

# saving this in case we need it later
stories_avg_score.to_csv("stories_avg_score.csv")

