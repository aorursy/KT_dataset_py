# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.list_tables()
hacker_news.table_schema('full')
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
"""
* How many stories (use the "id" column) are there of each type (in the "type" column) in 
the full table?
"""

query = """
SELECT CONCAT('The type "',type,'" has ',CAST(count(id) as string),' stories.') AS Result
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""
#hacker_news.estimate_query_size(query)

hacker_news.query_to_pandas(query)

"""* How many comments have been deleted? (If a comment was deleted the "deleted" column
in the comments table will have the value "True".)
"""
query = """SELECT CONCAT(CAST(COUNT(deleted) AS string),' comments have been deleted.') AS Result2
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            """
hacker_news.estimate_query_size(query)
hacker_news.query_to_pandas_safe(query)
            
"""* **Optional extra credit**: read about [aggregate functions
other than
COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) 
and modify one of the queries you wrote above to use a different aggregate function."""

query = """SELECT CONCAT(CAST(COUNTIF(deleted = True) AS string),' comments have been deleted.') AS BonusLeanring
            FROM `bigquery-public-data.hacker_news.full`
        """

hacker_news.query_to_pandas(query)

#Visualizing the data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot

new=new.convert_objects(convert_numeric=True)
#sns.lmplot("AvgSpeed", "Max5Speed", new)
#plot.figure(figsize = (20, 6))
sns.lmplot("Parent","ID",new)