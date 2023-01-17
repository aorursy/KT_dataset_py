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
#this was the change I overlooked. In the tutorial its a comments tabel but, here its a "full" table
hacker_news.head("full")
# Your code goes here :) # dont need HAVING here
QUERY2 = """
        SELECT type, COUNT(id)
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        """

# setting up the data frame and safely access the table and pandas library then run the df

type_counts = hacker_news.query_to_pandas_safe(QUERY2)
type_counts

#with no aggregates we get the f0_ which is the COUNT of the unique types. Here we have 5 types.
#Now lets tackle the deleted comments
#We have to use the "comments" table....I am confused as to why we are 
#switiching back and forth between these tables and when its right to do that..?
# after selecting the appropiate table we look at counting up the id to the deleted column with True values

hacker_news.head("comments")
#setup the sql request
QUERY2A = """
            SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
            """
#build the df
deleted_comments= hacker_news.query_to_pandas_safe(QUERY2A)
deleted_comments