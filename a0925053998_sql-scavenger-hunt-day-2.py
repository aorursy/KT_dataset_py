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
hacker_news.head("full")
hacker_news.head("stories")
hacker_news.head("full_201510")
query1 = """SELECT type, COUNT(id) as Quantity
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
         """
type_numbers = hacker_news.query_to_pandas_safe(query1)
type_numbers
comment1 = """SELECT type,deleted,COUNT(id) as del_num1
              FROM `bigquery-public-data.hacker_news.full`
              WHERE deleted = True
              GROUP BY type,deleted
              HAVING type='comment'
           """
CommentAllDeleted = hacker_news.query_to_pandas(comment1)
CommentAllDeleted.head()
comment2 = """SELECT type,deleted,COUNT(id) as Del_Quantity
              FROM `bigquery-public-data.hacker_news.full`
              WHERE deleted = True AND type='comment'
              GROUP BY type,deleted
           """
comments_deleted = hacker_news.query_to_pandas(comment2)
comments_deleted
ex_AF = """SELECT COUNT(id) as Total_ID, AVG(score) as Mean_Score,
                  MAX(score) as Max_Score, MIN(score) as Min_Score, SUM(score) as Total_Score
           FROM `bigquery-public-data.hacker_news.full`
        """
Ex_AggregateFunction = hacker_news.query_to_pandas(ex_AF)
Ex_AggregateFunction.head()
type_numbers.to_csv("StoriesQuantityInEveryType_Use_HackerNewsDataset.csv")
comments_deleted.to_csv("DeletedCommentsQuantity_Use_HackerNewsDataset.csv")
Ex_AggregateFunction.to_csv("AggregateFunctionExercise_Use_HackerNewsDataset.csv")