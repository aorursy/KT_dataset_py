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
# How many stories?
#print(hacker_news.head("full"))
print("\nNumber of Stories")
query_stories = """SELECT type, COUNT(id) as story_count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_count = hacker_news.query_to_pandas_safe(query_stories)
print(story_count)

# Deleted Comments
#print(hacker_news.head("comments"))
print("\nCount of Deleted Comments")
query_deleted_comments = """SELECT deleted, COUNT(id) as deleted_count
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
deleted_comments = hacker_news.query_to_pandas_safe(query_deleted_comments)
print(deleted_comments)

# Optional ... find the highest 'score' in comments for each type
print("\nHighest scoring comments by type")
query_score = """SELECT type, MAX(score) as Max_Score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE score > 0
            GROUP BY type
        """
max_scores = hacker_news.query_to_pandas_safe(query_score)
print(max_scores)