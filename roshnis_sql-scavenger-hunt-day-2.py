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
hacker_news.head("full")


#count the number of posts in each type like story, comment etc
query = """SELECT type,count(ID) as total_post
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

#running the query and storing the result to pandas dataframe
total_post = hacker_news.query_to_pandas_safe(query)

#displaying the first few of the result
total_post.head()
#listing all the post types
total_post
#view the table schema and values by printing first few rows
hacker_news.head("comments")
#selecting all the comments with deleted value as 'true'
query = """SELECT count(id) as deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted=True            
        """
#running the query and storing the result to pandas dataframe
deleted_comments = hacker_news.query_to_pandas_safe(query)

#displaying the first few of the result
deleted_comments.head()

deleted_comments
#editing the query using aggregate function sum(). since the deleted col has boolean value, 
#we can type cast it to int, it will convert True to 1, sum the values having deleted=true
query = """SELECT SUM(CAST(deleted as INT64)) as deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted=True
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)

#displaying the first few of the result
deleted_comments