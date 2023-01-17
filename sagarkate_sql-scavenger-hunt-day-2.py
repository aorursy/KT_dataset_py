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
import bq_helper
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
queryTypeOfStories = """
SELECT type, count(id) as num_of_stories
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""

resultTypeOfStories = hacker_news.query_to_pandas_safe(queryTypeOfStories)

print("Total no. of stories per type:")
print("type\tnum_of_stories")
for index, row in resultTypeOfStories.iterrows():
    print(row[0] + "\t" + str(row[1]))

#How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
queryDeletedComments = """
SELECT COUNT(1) as num_of_deleted_comments
FROM `bigquery-public-data.hacker_news.full`
WHERE deleted = True
"""

resultDeletedComments = hacker_news.query_to_pandas_safe(queryDeletedComments)

for index, row in resultDeletedComments.iterrows():
    print("Total no. of deleted comments: " + str(row[0]))