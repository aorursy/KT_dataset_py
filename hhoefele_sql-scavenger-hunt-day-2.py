#import Kaggle's BigQuery Python package: bq_helper
import bq_helper

#create helper object for the bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                       dataset_name = "hacker_news")

#print list of all tables in the hacker_news dataset
hacker_news.list_tables()
# print the first few rows of the "comments" table
hacker_news.head("comments")
# print table schema for "comments" table
hacker_news.table_schema("comments")
# create query 
query = """ SELECT parent, COUNT(id) 
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id)>10
"""
# check query size
hacker_news.estimate_query_size(query)
# create safe dataframe
popular_stories = hacker_news.query_to_pandas_safe(query)

# view first few rows of dataframe
popular_stories.head()
# save dataframe to csv file
popular_stories.to_csv("popular_stories.csv")
# print the first few rows of the "full" table
hacker_news.head("full")
# create query #2
query2 = """ SELECT type, COUNT(id) 
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING type = "story"
"""
# create safe dataframe
unique_stories = hacker_news.query_to_pandas_safe(query2)

# view first few rows of dataframe
unique_stories.head()
# create query #3
query3 = """ SELECT type, COUNT(deleted) 
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = TRUE
            GROUP BY type
            HAVING type = "comment"
        """
# create safe dataframe
deleted_stories = hacker_news.query_to_pandas_safe(query3)

# view first few rows of dataframe
deleted_stories.head()
# print dataframe results
print(deleted_stories)
# create query #4 
query4 = """  SELECT ht.type, ht.title, ht.id
            FROM `bigquery-public-data.hacker_news.full` ht,
            
            (SELECT max(id) AS max_id, type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type 
            HAVING type = "story") maxid
            
            WHERE ht.type = maxid.type AND ht.id = maxid.max_id;
        """
# check query size
hacker_news.estimate_query_size(query4)
# create safe dataframe
latest_story = hacker_news.query_to_pandas_safe(query4)
# view first few rows of dataframe
latest_story.head()
# save dataframe to csv file
latest_story.to_csv("latest_story.csv")
