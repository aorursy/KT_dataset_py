# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# Have a look at the table "full"
hacker_news.head("full")


# Check out the schema. See if there is a defined list of types.
hacker_news.table_schema("full")
# Get the count of stories by type
query = """
    SELECT type, count(ID)
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
    """
stories_by_type = hacker_news.query_to_pandas_safe(query)
# Since there are only six types, a table is probably the best output
stories_by_type
query = """
    SELECT deleted, count(ID)
    FROM `bigquery-public-data.hacker_news.full`
    WHERE type = "comment"
    GROUP BY deleted
    """
comments_by_deleted = hacker_news.query_to_pandas_safe(query)
comments_by_deleted
query = """
    SELECT `by`, AVG(score), count(ID)
    FROM `bigquery-public-data.hacker_news.full`
    WHERE NOT IS_NAN(score)
    GROUP BY `by`
    HAVING count(ID) > 10
    """
average_scores_by_popular_authors = hacker_news.query_to_pandas_safe(query)
average_scores_by_popular_authors.head()
average_scores_by_popular_authors.describe()