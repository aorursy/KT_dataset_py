# import package with helper functions 
import bq_helper as bq

# create a helper object for this dataset
hacker_news = bq.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")

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
#Task1: How many stories (use the "id" column) are there of each type (in the "type" column) in the full t
#Looking at the schema of full table:

hacker_news.table_schema("full")

#the type column has the following values: comment, comment_ranking, poll, story, job, pollopt
#Task1 Query
one = """SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full` 
GROUP BY type """
#Solution
hacker_news.query_to_pandas_safe(one).head()

#Insight: comments rule the roost!
hacker_news.head("comments", selected_columns = "deleted", num_rows = 20)
#Task2: How many comments have been deleted?
two = """SELECT COUNT(id) FROM `bigquery-public-data.hacker_news.comments`GROUP BY deleted"""

hacker_news.query_to_pandas_safe(two)

#Insight: 227736 comments have been deleted. 
#Double checking the above code
#Why?...because sometimes 0s and 1s are coded unlike what we may assume.
twoo = """SELECT COUNT(id) FROM `bigquery-public-data.hacker_news.comments` WHERE deleted = True"""

hacker_news.query_to_pandas_safe(twoo)
#Its safe to say my insight was true. 
#Task3: using other aggregate function
#I would like to know the total time spent on each type in the full table

three = """SELECT type, SUM(time) FROM `bigquery-public-data.hacker_news.full` GROUP BY type """

hacker_news.query_to_pandas_safe(three)

#Insight: Comment wins again! 