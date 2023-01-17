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
print("How many stories are there of each type in the full table?")

query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
group_by_type = hacker_news.query_to_pandas_safe(query)
group_by_type
print("How many comments have been deleted?")

query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
group_by_status = hacker_news.query_to_pandas_safe(query)
group_by_status
print("What's the highest score by type ?")

query = """SELECT type, MAX(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
max_score_by_type = hacker_news.query_to_pandas_safe(query)
max_score_by_type
print("Author with highest Score")

query = """SELECT full_table.by as Author, SUM(score) as Score
            FROM `bigquery-public-data.hacker_news.full` as full_table
            GROUP BY full_table.by
            HAVING full_table.by != 'NaN' and Score >= 0
            ORDER BY Score DESC LIMIT 1
        """
author_with_highest_score = hacker_news.query_to_pandas_safe(query)
author_with_highest_score.head()