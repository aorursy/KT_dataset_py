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

#Let's take a look at the 'Full' table(I am in a little confusion whether this was the right kind of table to use so bear with me)
hacker_news.head("full")

#Let us find out the number of stories of each kind
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
all_types = hacker_news.query_to_pandas_safe(query1)
all_types
#There appear to be 5 types of stories with their corresponding counts.

#How about the number of comments that were deleted? We go on working with the 'deleted' column. 
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = TRUE
            GROUP BY deleted
          """

deleted_count = hacker_news.query_to_pandas_safe(query2)
deleted_count
#The number of deleted comments is 227736!

# Optional Extra Credit
#What is the average for the column 'dead' from the 'stories' table?
hacker_news.head("stories")
query3 = """SELECT dead, AVG(id)
            FROM `bigquery-public-data.hacker_news.stories`
            GROUP BY dead
          """  
dead_avg = hacker_news.query_to_pandas_safe(query3)
dead_avg
#Apparently a very large number! Is this even correct? Am I missing something?