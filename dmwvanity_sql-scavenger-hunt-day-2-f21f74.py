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
# How many stories are there of each type in the full table?
# Let's have a look at the table first and print the first couple rows of the "full" table
hacker_news.head("full")

# Now write the query
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# Run the safe query
diff_types = hacker_news.query_to_pandas_safe(query)
# View the dataframe
diff_types
# 'comment' makes up 13,487,115/16,356,047 or 82.5% of the types of articles in Hacker News. 
# How many comments have been deleted?
# As we saw earlier, the comments table contains information about comment deletions. Let's take another look at the table.
hacker_news.head("comments")
# Now write the query returning only the count of the number of comments having been deleted
# Since the comments table contains all article types that are comments, and contains the information related to whether the comment was deleted, we query the comments table
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
# Run the query in safe mode
deleted_comments = hacker_news.query_to_pandas_safe(query)
# View the dataframe
deleted_comments
# I am wondering what are the minimum, maximum and average comment rankings.
query = """SELECT AVG(ranking) AS avg_rank, MIN(ranking) AS min_rank, MAX(ranking) AS max_rank
            FROM `bigquery-public-data.hacker_news.comments`
        """
# Run the query in safe mode
ranks = hacker_news.query_to_pandas_safe(query)
# View the dataframe
ranks
# How many rows in the table?
query = """SELECT COUNT(*)
            FROM `bigquery-public-data.hacker_news.comments`
        """
# Run the query in safe mode
rows = hacker_news.query_to_pandas_safe(query)
# View the dataframe
rows
# It appears there are many rankings equal to zero because the average ranking is so lo. We can find this out with a new query
query = """SELECT ranking, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY ranking
            HAVING count > 10000
            ORDER BY ranking
        """
# Run the query in safe mode
rankings_count = hacker_news.query_to_pandas_safe(query)
# View the dataframe
rankings_count
# 4,165,946/8,399,417 or 50% of the comments have zero ranking.