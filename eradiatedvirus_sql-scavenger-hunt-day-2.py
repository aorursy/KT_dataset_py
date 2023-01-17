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
hacker_news.table_schema("full")


hacker_news.head("full")
query1 = """ SELECT type, count(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

#check size before running the query
hacker_news.estimate_query_size(query1)

stories_by_type = hacker_news.query_to_pandas_safe(query1)
stories_by_type.head()
hacker_news.list_tables()
# since all of there filtering can be done before aggregating, the filtering
# can be done as a WHERE. GROUP BY and HAVING is unecessary 
query2 = """ SELECT count(id) as cnt
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'comment' AND deleted = True
        """
# achieves the same as above but with HAVING instead
query3 = """ SELECT type, count(id) as cnt
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            GROUP BY type
            HAVING type = 'comment'
        """


#check size before running the query
hacker_news.estimate_query_size(query2)

hacker_news.estimate_query_size(query3)
deleted_comments_1 = hacker_news.query_to_pandas_safe(query2)
print(deleted_comments_1.head())

deleted_comments_2 = hacker_news.query_to_pandas_safe(query3)
print(deleted_comments_2.head())
# gets the average score for each author ignoring NaN values where 
# the author has more than 3 scores
query4 = """ SELECT `by`, AVG(score) as Avg_Score, COUNT(score) as cnt
            FROM `bigquery-public-data.hacker_news.full`
            WHERE score IS NOT NULL
            GROUP BY `by`
            HAVING COUNT(score) > 1
        """
hacker_news.estimate_query_size(query4)
average_Scores = hacker_news.query_to_pandas_safe(query4)
average_Scores.head()
average_Scores.sort_values(['Avg_Score', 'by'], ascending=[0, 1]).head()