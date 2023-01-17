# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head('full')
# query to pass to BQ
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
type_story = hacker_news.query_to_pandas_safe(query)
type_story.head()
# query to pass to BQ
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            GROUP BY type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted_comments = hacker_news.query_to_pandas_safe(query)

deleted_comments.head()
# query to pass to BQ
query = """SELECT type, AVG(LENGTH(`title`))
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
title_length = hacker_news.query_to_pandas_safe(query)
title_length.head()
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
plt.figure(figsize=(15,5))
sns.barplot(x="type", y="f0_", data=title_length)
plt.xlabel("Type post")
plt.ylabel("Average length title")