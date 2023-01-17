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
# Like yesterday we start by importing bigQueryhelper
import bq_helper

# Initiating an instance of bigQueryhelper to look at the hacker news database
HackerNews = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# Let's look at how many tables are there in this databse
HackerNews.list_tables()
# Unlike yesterday's airquality database that had only one table, the hacker news database
# has 4 tables, namely - comments, full, full201510 and stories.
# For the first question in the scavenger hunt, we're going to be looking in the full table.
HackerNews.head('full')

# Let's start building the first SQL query
query1 = """ SELECT type, COUNT (id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY type """
number_of_stories = HackerNews.query_to_pandas_safe(query1)

print(number_of_stories.head())
# Let's move on to the second query for today's challenge.
# For this query, we're going to be using the comments table in the database
HackerNews.head('comments')
query2 = """ SELECT deleted, COUNT (id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted"""
number_of_comments = HackerNews.query_to_pandas_safe(query2)
# Let's check to make sure that our query actually pulled out the data we needed
print (number_of_comments)
# As we can see, there ar eonly two possible outcomes True and None, so we subset the data
# frame to get the number of comments that were deleted.
print(number_of_comments[number_of_comments.deleted == True])