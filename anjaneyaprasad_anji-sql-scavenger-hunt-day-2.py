import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.head("comments")
hacker_news.table_schema("full")
query = """SELECT parent, COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
           HAVING COUNT(id) > 10
        """

popular_stories = hacker_news.query_to_pandas_safe(query)

popular_stories.head()
#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query = """SELECT type, count(id)
             FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_count = hacker_news.query_to_pandas_safe(query)
type_count.head()
#How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# deleted column is of type BOOL.
# SchemaField('deleted', 'boolean', 'NULLABLE', 'Is deleted?', ()),

query ="""SELECT count(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted IS TRUE 
        """

deleted_count = hacker_news.query_to_pandas_safe(query)
deleted_count.head()
# Average ranking of an Author

query ="""SELECT author, AVG(ranking)
            FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY author
           ORDER BY 2 DESC
        """

avg_ranking = hacker_news.query_to_pandas_safe(query)
avg_ranking.head()