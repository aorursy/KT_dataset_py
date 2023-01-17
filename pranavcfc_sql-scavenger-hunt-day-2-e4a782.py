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
# Your code goes here :)
from google.cloud import bigquery
client = bigquery.Client()

 
query = client.query("""select count(id), type
            FROM `bigquery-public-data.hacker_news.full`
            group by type,id
            
        """)
print(query.result().to_dataframe())


query2 = client.query("""select count(type) as comments_deleted
            FROM `bigquery-public-data.hacker_news.full`
            where type="comment"
            and deleted = true
            """)
print(query2.result().to_dataframe())
query3 = client.query("""select type, sum(id)
            over (order by type) as type_comments 
            FROM `bigquery-public-data.hacker_news.full`
            where type="comment"
            and deleted = true
""")
print(query3.result().to_dataframe())