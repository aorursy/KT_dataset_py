# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")


# print the first couple rows of the "full" table
hacker_news.head("full")
# How many stories for each type?  
query1 = """SELECT type, COUNT(id) as StoryCount
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY StoryCount
        """
result1 = hacker_news.query_to_pandas_safe(query1)
print(result1)
# print the first couple rows of the "comments" table
hacker_news.head("comments")
# How many stories for each type?  
query2 = """SELECT deleted, COUNT(id) as DeletedCount
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            
        """
result2 = hacker_news.query_to_pandas_safe(query2)
print(result2)
# Average Score For Each Story Type?  
query1a = """SELECT type, AVG(score) as AverageScore
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY AverageScore DESC
        """
result1a = hacker_news.query_to_pandas_safe(query1a)
print(result1a)
# Average Ranking for Deleted Status?  
query3 = """SELECT deleted, AVG(ranking) as AverageRanking
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            
        """
result3 = hacker_news.query_to_pandas_safe(query3)
print(result3)