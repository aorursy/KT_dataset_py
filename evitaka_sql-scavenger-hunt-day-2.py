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
# print the first couple rows of the "full" table
hacker_news.head("full")
# form the query for question 1
query2 = """SELECT type, COUNT(id) AS num
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY num DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
stories_per_type = hacker_news.query_to_pandas_safe(query2)

# view the 5 first rows of the dataframe
stories_per_type.head()
stories_per_type.to_csv("stories_per_type.csv")
# form the query for question 2
query3 = """SELECT COUNT(id) as deletedNum
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
nom_of_deleted_comments = hacker_news.query_to_pandas_safe(query3)

# view the 5 first rows of the dataframe
nom_of_deleted_comments.head()
# form the query for question 3
query4 = """SELECT type, AVG(score) as avgScore
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY avgScore DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
avg_score_per_type = hacker_news.query_to_pandas_safe(query4)

# view the 5 first rows of the dataframe
avg_score_per_type.head()
import matplotlib.pyplot as plt
ax = avg_score_per_type.plot(x=avg_score_per_type.type,rot=0,kind='bar', title ="Average score of comments' types")
ax.set_xlabel("Comments' Types",fontsize=12)
ax.set_ylabel("Average Score",fontsize=12)