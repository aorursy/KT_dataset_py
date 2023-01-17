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

# print the first couple rows of the "comments" table
hacker_news.head("comments")

# print all the tables in this dataset (there's only one!)
hacker_news.list_tables() 

# query to pass to 
# query = """SELECT parent, COUNT(id)
#            FROM `bigquery-public-data.hacker_news.comments`
#            GROUP BY parent
#            """
#countids =hacker_news.query_to_pandas_safe(query)


#print(countids)

query2 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'story'
            GROUP BY type
            
            """
stories =hacker_news.query_to_pandas_safe(query2)


print(stories)

query3 = """SELECT author, count (id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = true
            GROUP BY author
            
            """
deleted =hacker_news.query_to_pandas_safe(query3)


print(deleted)
