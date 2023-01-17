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
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# Q1: How many stories (use the "id" column) are 
#there of each type (in the "type" column) in the full table?

MyQuery1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

p1 = hacker_news.query_to_pandas_safe(MyQuery1)
    
a1 = p1.type.unique() 
a1

# Answer: 5 type of coomments, they are 'comment', 'story', 'pollopt', 'job', 'poll'


# Q2: How many comments have been deleted?
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)

MyQuery2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """

p2 = hacker_news.query_to_pandas_safe(MyQuery2)

p2.head()

# answer: 227,736 comments have been deleted yey :v


# Extra Credit
MyQuery3 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """

p3 = hacker_news.query_to_pandas_safe(MyQuery3)

p2.head()