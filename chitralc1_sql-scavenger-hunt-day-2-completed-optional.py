# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
query1 = """SELECT type, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

count_type = hacker_news.query_to_pandas_safe(query1)
count_type
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.barplot(count_type['type'], count_type['count'])
query2 = """SELECT COUNT(deleted) AS count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
count_deleted = hacker_news.query_to_pandas_safe(query2)
count_deleted
print("Deleted Comments:", count_deleted['count'][0])
query3 = """SELECT author, MAX(id) AS MaxID
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
        """
max_id = hacker_news.query_to_pandas_safe(query3)
max_id
count_type.to_csv('count_type.csv')
count_deleted.to_csv('deleted.csv')
max_id.to_csv('max.csv')
