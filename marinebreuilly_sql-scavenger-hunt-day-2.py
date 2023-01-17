# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print a list of all the tables in the hacker_news dataset
print(hacker_news.list_tables())

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """ SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`   
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default (choosed 0.2GB)
popular_stories = hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.2)

popular_stories.head()
# 1 - How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
print("Q1: ")
# print the first couple rows of the "full" table
hacker_news.head("full")
# query to pass to
query1 = """ SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`   
            GROUP BY type
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default (choosed 0.3GB)
stories_type_nb = hacker_news.query_to_pandas_safe(query1, max_gb_scanned=0.3)
for i in range(len(stories_type_nb.type)):
    print('There are {1} stories of type {0};'.format(stories_type_nb.type[i], stories_type_nb.f0_[i]), end="\n")

# 2 - How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
print("\n", "Q2: ")
# print the first couple rows of the "full" table
hacker_news.head("comments")
# query to pass to
query2 = """ SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default (choosed 0.2GB)
comments_deleted_nb = hacker_news.query_to_pandas_safe(query2, max_gb_scanned=0.2)
for i in range(len(comments_deleted_nb.deleted)):
    if comments_deleted_nb.deleted[i] == True:
        print("{0} comments have been deleted from the Hacker News dataset.".format(comments_deleted_nb.f0_[i]))
