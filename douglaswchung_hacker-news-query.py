# import package with helper functions 

import bq_helper



# create a helper object for this dataset

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="hacker_news")



# print the first couple rows of the "comments" table

hacker_news.head("comments")
# Your Code Here

query = """ SELECT type, COUNT(id)

            FROM `bigquery-public-data.hacker_news.full`

            GROUP BY type

"""

stories_of_each_type = hacker_news.query_to_pandas_safe(query)

stories_of_each_type.head()
# Your Code Here

query = """ SELECT deleted, COUNT(id)

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY deleted

"""

num_deleted_comments = hacker_news.query_to_pandas_safe(query)

num_deleted_comments.head()
# Your Code Here

query = """ SELECT type, MAX(time)

            FROM `bigquery-public-data.hacker_news.full`

            GROUP BY type

"""

latest_time_of_each_type = hacker_news.query_to_pandas_safe(query)

latest_time_of_each_type.head()