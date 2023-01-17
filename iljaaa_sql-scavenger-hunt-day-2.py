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
query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
        """
hacker_news.estimate_query_size(query)
count_id = hacker_news.query_to_pandas_safe(query)
count_id
query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
hacker_news.estimate_query_size(query)
deleted = hacker_news.query_to_pandas_safe(query)
deleted
query = """SELECT author, COUNT(id), AVG(ranking)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            ORDER BY AVG(ranking) DESC
        """
hacker_news.estimate_query_size(query)
avg_ranking = hacker_news.query_to_pandas_safe(query)
avg_ranking.head()
query = """SELECT author, COUNT(id), AVG(ranking)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE author != 'None'
            GROUP BY author
            HAVING COUNT(id) > 10
            ORDER BY AVG(ranking) DESC
        """
hacker_news.estimate_query_size(query)
avg_ranking = hacker_news.query_to_pandas_safe(query)
avg_ranking.head()
import matplotlib.pyplot as plt
plt.scatter(avg_ranking['f0_'], avg_ranking['f1_'])
avg_ranking[avg_ranking['f0_']>15000]
filter_ = avg_ranking['f0_']<1000
x = avg_ranking[filter_]['f0_']
y = avg_ranking[filter_]['f1_']
plt.scatter(x, y)