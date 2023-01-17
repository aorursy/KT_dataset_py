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
#what tables are there in the dataset
hacker_news.list_tables()
# could also see this by looking at the data tab here on Kaggle to the right
#this question is asking about the "full" table. Let's look at the first few rows of the full table
# print the first couple rows of the "full" dataset
hacker_news.head("full")
#I see the column "type" that the question is asking about
#Let's make the query
query_numtype_infull = """SELECT type, COUNT(ID)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
#run the query safely (with an eye to limiting how much data I use against my quota) and get a dataframe
numtype_full = hacker_news.query_to_pandas_safe(query_numtype_infull)

# save our dataframe as a .csv 
numtype_full.to_csv("numtype_full.csv")
# print the first couple rows of the "comments" table
hacker_news.head("comments")
# The deleted column will tell me if a comment has been deleted
#Let's make the query
query_numdel_incomments = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
"""
#run the query safely (with an eye to limiting how much data I use against my quota) and get a dataframe
numdel_incomments = hacker_news.query_to_pandas_safe(query_numdel_incomments)

# export dataframe as a .csv 
numdel_incomments.to_csv("numdel_incomments.csv")
# I'm going to answer the question: What is the average ranking of deleted comments?
#Let's make the query
query_avgranking_delcomments = """SELECT deleted, AVG(ranking)
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY deleted
"""
#run the query safely (with an eye to limiting how much data I use against my quota) and get a dataframe
avgrank_delcomments = hacker_news.query_to_pandas_safe(query_avgranking_delcomments)

# export dataframe as a .csv 
avgrank_delcomments.to_csv("avgrank_delcomments.csv")