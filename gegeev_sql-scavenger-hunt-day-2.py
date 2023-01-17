# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.list_tables()
hacker_news.head("full")
# to answer How many stories have each type 
query_many_stories = """SELECT type, COUNT(*)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
hacker_news.query_to_pandas_safe(query_many_stories)
# query to find deleted items
query_del = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
        """
df_del=hacker_news.query_to_pandas_safe(query_del)
df_del
print ('there is ' + str(df_del.loc[0][1]) + ' deleted items in dataset')
# query to use max & min
query_maxmin_type = """SELECT MAX(type) as max, MIN(type) as min
            FROM `bigquery-public-data.hacker_news.full`
           """
df_mm_type=hacker_news.query_to_pandas_safe(query_maxmin_type)
df_mm_type.head()
