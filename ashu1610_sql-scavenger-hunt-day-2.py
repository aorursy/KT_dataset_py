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
stories_by_type = """SELECT type, count(id)
             from `bigquery-public-data.hacker_news.full`
             group by type"""

same_type_stories = hacker_news.query_to_pandas_safe(stories_by_type)

same_type_stories.head()
query_2 = """SELECT deleted, count(id)
             from `bigquery-public-data.hacker_news.comments`
             group by deleted
             having deleted = true
             """

del_comm = hacker_news.query_to_pandas_safe(query_2)

del_comm.head()
query_3 = """
select
  sum(
    case
      when deleted = True
      then 1
      else 0
    end
  ) as deleted_comments
from
  `bigquery-public-data.hacker_news.comments`
where
  deleted = True
"""

credit_query = hacker_news.query_to_pandas_safe(query_3)

credit_query.head()
