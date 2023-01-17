import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")

#Q1 How many stories (use the "id" column)
#are there of each type (in the "type" column) in the full table?

#create the query
query_nb_stories ="""SELECT type, COUNT(id) FROM `bigquery-public-data.hacker_news.full` GROUP BY type"""


#load query result to pandas df
number_stories = hacker_news.query_to_pandas_safe(query_nb_stories)
#print the result
number_stories
#Q2: How many comments have been deleted? (If a comment was deleted 
#the "deleted" column in the comments table will have the value "True".)

#create the query
query_nb_deleted ="""SELECT deleted, COUNT(id) FROM `bigquery-public-data.hacker_news.full` GROUP BY deleted"""



#load query result to pandas df
number_deleted = hacker_news.query_to_pandas_safe(query_nb_deleted)
#print out the df
number_deleted
#Q3 Optional extra credit: read about aggregate functions other than COUNT() and modify one of 
#the queries you wrote above to use a different aggregate function.

#I will aggregate the max id number of each type.

#create the query
query_max_nbstories ="""SELECT type, max(id) FROM `bigquery-public-data.hacker_news.full` GROUP BY type"""


#load query result to pandas df
number_max_nbstories= hacker_news.query_to_pandas_safe(query_max_nbstories)
#print out the df
number_max_nbstories