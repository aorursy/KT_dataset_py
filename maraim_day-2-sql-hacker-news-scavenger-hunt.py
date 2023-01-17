# 1. import package with helper functions 
import bq_helper

# 2. create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# 3. Examine the dataspace¶
hacker_news.list_tables()

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
# 1. number of stories per type
num_stories_per_type_query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
num_stories_per_type = hacker_news.query_to_pandas_safe(num_stories_per_type_query)
num_stories_per_type
# 2. number of deleted comments
# I was not sure in which table so I am querying both tables. 
def num_deleted_comments(table_name):
    return """SELECT count(id)
            FROM `bigquery-public-data.hacker_news.{0}`
            GROUP BY deleted
            HAVING deleted = True
        """.format(table_name)

num_deleted_comments_comments_table = hacker_news.query_to_pandas_safe(num_deleted_comments('comments'))
num_deleted_comments_full_table = hacker_news.query_to_pandas_safe(num_deleted_comments('full'))

print("Number of deleted comments in comments table: ", num_deleted_comments_comments_table)
print("Number of deleted comments in full table: ", num_deleted_comments_full_table)
alt_query = """SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            """
deleted_comments = hacker_news.query_to_pandas_safe(alt_query)
deleted_comments.head()
query3 = """SELECT sum(if (deleted = True,1,0)) num_deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
        """
hacker_news.estimate_query_size(query3)
num_deleted_comments_q3 = hacker_news.query_to_pandas_safe(query3)
display(num_deleted_comments_q3.head())
num_deleted_comments_q3
# set up a variable for the repeated portion of the space/table name:
table = "bigquery-public-data.hacker_news.full"

query = """
        SELECT type, AVG(score) avg_score
        FROM `{}`
        GROUP BY type
        ORDER BY avg_score DESC
""".format(table)

# Check the usage
display(hacker_news.estimate_query_size(query))

avg_scors = hacker_news.query_to_pandas_safe(query)

# Check out the results
avg_scors.columns = ['type', 'average_score']
display(avg_scors)
import matplotlib.pyplot as plt
%matplotlib inline

colors=['red','blue','yellow','green','pink']
plt.style.use('ggplot')
plt.bar(avg_scors['type'], avg_scors['average_score'], color=colors)
