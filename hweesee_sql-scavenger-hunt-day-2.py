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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print all the tables in this dataset
hacker_news.list_tables()
# print the first couple rows of the "full" table
hacker_news.head("full")
# query to count the items uniquedly identify by "id" group by "type" in "full" table
query_Story = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# check how big this query will be
hacker_news.estimate_query_size(query_Story)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
storycount_bytype = hacker_news.query_to_pandas_safe(query_Story)
storycount_bytype
# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to count the items uniquedly identify by "id" group by "type" in "full" table
query_deleted_comment = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
# check how big this query will be
hacker_news.estimate_query_size(query_deleted_comment)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
deleted_comments = hacker_news.query_to_pandas_safe(query_deleted_comment)
deleted_comments
# query to count the items uniquedly identify by "id" group by "type" in "full" table
query_random_story = """SELECT ANY_VALUE(STORY_TYPE.type) AS any_story
            FROM (
            SELECT type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING COUNT(id) > 1000000
            ) STORY_TYPE
        """
# check how big this query will be
hacker_news.estimate_query_size(query_random_story)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
random_story = hacker_news.query_to_pandas_safe(query_random_story)
random_story
from google.cloud import bigquery

if __name__ == '__main__':
    client = bigquery.Client()
    query_job = client.query("""SELECT
                                  fruit,
                                  ANY_VALUE(fruit) OVER (ORDER BY LENGTH(fruit) ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS any_value
                                FROM UNNEST(["apple", "banana", "pear"]) as fruit; 
                            """)

    print(query_job.result().to_dataframe())