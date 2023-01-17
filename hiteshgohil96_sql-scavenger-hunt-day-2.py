# import package with helper functions 

import bq_helper



# create a helper object for this dataset

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="hacker_news")



# print the first couple rows of the "comments" table

hacker_news.head("comments")
# query to pass to 

query = """SELECT parent, COUNT(id) as counts

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY parent

            HAVING COUNT(id) > 10

        """
# the query_to_pandas_safe method will cancel the query if

# it would use too much of your quota, with the limit set 

# to 1 GB by default

popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# Your code goes here 



query = """ select type, count(id) as counts from `bigquery-public-data.hacker_news.full` group by type """

stories = hacker_news.query_to_pandas_safe(query)

print(stories)



query = """ select type,count(deleted) as deleted_count from `bigquery-public-data.hacker_news.full` where deleted = True and type = 'comment' group by type """



deleted = hacker_news.query_to_pandas_safe(query)

deleted

print('Total {} comments were deleted.'.format(deleted['deleted_count'][0]))

# EXTRA CREDIT



query = """ select type, avg(score) as avg_score from `bigquery-public-data.hacker_news.full` group by type having sum(score) > 0 """

score = hacker_news.query_to_pandas_safe(query)

score