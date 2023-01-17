# Your code goes here :)
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# How many stories are there of each type 
query = """SELECT type, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_types = hacker_news.query_to_pandas_safe(query)
story_types
# Your code goes here :)
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# How many comments have been deleted? 
query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            where deleted
        """
deleted_comments_count = hacker_news.query_to_pandas_safe(query)
deleted_comments_count.values[0][0], ' comments have been deleted'
# Your code goes here :)
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# How many comments of each type have been deleted? 
query = """SELECT type, COUNT(id) as number_deleted
            FROM `bigquery-public-data.hacker_news.full`
            where deleted
            GROUP BY type
        """
deleted_comments_count_by_type = hacker_news.query_to_pandas_safe(query)
deleted_comments_count_by_type
# Your code goes here :)
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# Optional extra credit: 
# What is the average ranking of the comments for each author. Show top twenty best ranked authors
query = """SELECT author, avg(ranking) as avg_ranking
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            order by avg_ranking desc
            limit 20
        """
avg_ranking_by_author = hacker_news.query_to_pandas_safe(query)
avg_ranking_by_author