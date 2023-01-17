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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")


query = """SELECT type, count(id) as num_of_articles
            FROM `bigquery-public-data.hacker_news.full`
            group by type
        """
temp = hacker_news.query_to_pandas_safe(query)
temp.head(20)
query = """SELECT  type, count(id) as deleted_full_tbl
            FROM `bigquery-public-data.hacker_news.full` where deleted = True
            group by type
        """
temp = hacker_news.query_to_pandas_safe(query)
temp.head()
query = """SELECT  count(id) as deleted_comments_tbl
            FROM `bigquery-public-data.hacker_news.comments`
            where deleted is True
        """
temp = hacker_news.query_to_pandas_safe(query)
temp.head()
query = """with tb1 as 
            (SELECT type, count(id) as num_of_articles
            FROM `bigquery-public-data.hacker_news.full`
            group by type)
            select avg(num_of_articles) as average_num_articles, 
            min(num_of_articles) as least, 
            max(num_of_articles) as most from tb1
        """
temp = hacker_news.query_to_pandas_safe(query)
temp.head()
query = """ 
            SELECT author, count(*) as num_articles, avg(ranking) as average_ranking
            from `bigquery-public-data.hacker_news.comments`
            where author is not null and ranking > 0 
            group by author having count(*) > 10
            order by avg(ranking) asc
        """

hacker_news.estimate_query_size(query)
comments_avg_ranking = hacker_news.query_to_pandas_safe(query)
comments_avg_ranking.head()
comments_avg_ranking.info()
from matplotlib import pyplot as plt
plt.scatter(comments_avg_ranking['num_articles'],comments_avg_ranking['average_ranking'])