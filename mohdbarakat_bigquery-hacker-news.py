# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# explore the data set
hacker_news.head("comments")
query = """
        SELECT parent, COUNT(id) as count
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY parent
        HAVING count(id) > 10
        """

popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# exploring the Full table
hacker_news.head("full")
query = """
        SELECT type, COUNT(id) as count
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        HAVING type = 'story'
        """
stories_count = hacker_news.query_to_pandas_safe(query)
stories_count
query = """
        SELECT deleted, COUNT(id) as count
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY deleted
        HAVING deleted = TRUE
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments

query = """
        SELECT parent as Comment_ID, max(ranking) as Max_Ranking
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY Comment_ID
        ORDER BY Max_Ranking DESC
        """

MaxRanking = hacker_news.query_to_pandas_safe(query)
MaxRanking.head()