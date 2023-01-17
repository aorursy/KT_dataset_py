# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
#hacker_news.head("comments")

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

print('How many stories  are there of each type  full table?')

query_count_by_type = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
#df_count_by_type=hacker_news.query_to_pandas_safe(query_count_by_type)
#hacker_news.estimate_query_size(query_count_by_type)
#df_count_by_type=
print(hacker_news.query_to_pandas(query_count_by_type))


print('How many comments have been deleted?')
#query_count_deleted = """SELECT  id,by FROM `bigquery-public-data.hacker_news.comments` """
query_count_deleted = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted is True
        """

#hacker_news.estimate_query_size(query_count_deleted)
print(hacker_news.query_to_pandas(query_count_deleted))


print('How many stories  are there of each type  full table? --using a different aggregate function other than  COUNT')
query_count_by_type="""SELECT type, sum(1)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
print(hacker_news.query_to_pandas(query_count_by_type))


print('How many comments have been deleted? --using a different aggregate function other than  COUNT')
#query_count_deleted = """SELECT  id,by FROM `bigquery-public-data.hacker_news.comments` """
query_count_deleted = """SELECT COUNTIF(deleted is True)
            FROM `bigquery-public-data.hacker_news.comments`
        """
#hacker_news.estimate_query_size(query_count_deleted)
print(hacker_news.query_to_pandas(query_count_deleted))



#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.