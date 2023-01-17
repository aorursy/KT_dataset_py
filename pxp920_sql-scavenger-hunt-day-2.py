#### DAY 2 SCAVENGER HUNT ####
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.list_tables()
# First 5 rows of "full" table
hacker_news.head("full")
# "How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?"
# Question is worded a bit poorly with the "id" and "type" indicators

query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

full_types = hacker_news.query_to_pandas_safe(query)
full_types.head()
# First 5 rows of "full" table
hacker_news.head("comments")
# First 5 rows of "comments" table
hacker_news.head("comments")
# "How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)"
# Ensure you're pointing to the 'comments' table

query = """select count(*) as NumberOfDeletedComments
           from `bigquery-public-data.hacker_news.comments` 
           where deleted=True
        """

hacker_news.query_to_pandas_safe(query)
# "Modify one of the queries you wrote above to use a different aggregate function."
# I will calculate Min/Max comments on a story posted by each of the users

# The first subquery will retrieve the distinct list of stories by each ID
# The second subquery will retrieve the number of comments by each story
# The top level query will join the two tables thus generating a by (author), story and number of comments
# field.  I further reduce this by grouping the max and min values for each author

query = """
           select `by`, max(NumberOfComments) as BestStory, min(NumberOfComments) as WorstStory
           from 
           (
               
               select distinct `by`, id
               from `bigquery-public-data.hacker_news.full` 
               where type='story'
           ) a     
           join
           (
               select parent, count(*) as NumberOfComments
               from `bigquery-public-data.hacker_news.comments` 
               group by parent
           ) b
           on a.id = b.parent
           group by `by`
           order by max(NumberOfComments) desc
        """

hacker_news.query_to_pandas_safe(query)
