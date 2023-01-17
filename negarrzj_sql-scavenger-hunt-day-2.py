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
# Your code goes here :)

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# import our bq_helper package (from Kaggle)

import bq_helper 

# create a helper object for our bigquery dataset (helper project from Kaggle)

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")

# print a list of all the tables in the hacker_news dataset

hacker_news.list_tables()

hacker_news.head('comments')





#Example: Which Hacker News comments generated the most discussion?

most_discution_query = """ SELECT count(id) as count_id, parent

                            FROM `bigquery-public-data.hacker_news.comments` 

                            GROUP BY parent

                            HAVING count(id)>10 

                            ORDER BY count_id DESC """

#convert popular stories to CSV file

popular_stories = hacker_news.query_to_pandas_safe(most_discution_query)



popular_stories[['count_id','parent']]

                    

# writing query to find total stories of each type 

count_stories_each_type_query = """ SELECT count(id) as count_id, type

                            FROM `bigquery-public-data.hacker_news.full` 

                            GROUP BY type """

# query to pandas

stories_each_type = hacker_news.query_to_pandas_safe(count_stories_each_type_query)

# show result in a table

stories_each_type[['count_id','type']]
count_deleted_comments_query = """ SELECT count(id) as count_id, deleted

                                   FROM `bigquery-public-data.hacker_news.comments` 

                                   GROUP BY deleted

                                   HAVING deleted = TRUE """

 

# query to pandas and show result                                  

deleted_comments = hacker_news.query_to_pandas_safe(count_deleted_comments_query)

deleted_comments[['count_id','deleted']]
