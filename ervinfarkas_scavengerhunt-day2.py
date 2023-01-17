# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import our bq_helper package
import bq_helper 
# create a helper object for Hacker News dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print a list of all the tables in the Hacker News dataset
hacker_news.list_tables()
# check 'comments' table content
hacker_news.table_schema('comments')
# this query looks in the 'comments' table in the Hacker News
# dataset, then gets which parent id generated most comments (id) 
# For that I used a select with groupb by parent and count id and then order counted results as desc and select first row only.
query = """SELECT parent, count(ID) as related_comments
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            ORDER BY related_Comments DESC
            LIMIT 1
        """

# check how big this query will be
hacker_news.estimate_query_size(query)
# run the query and get parent id with most comments
mostCommentedId=hacker_news.query_to_pandas_safe(query)



#same but finding max in python - but limiting to groups having more than 200 comments
query = """SELECT parent, count(ID) as related_comments
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING count(ID) > 200
        """
Id=hacker_news.query_to_pandas_safe(query)

print(Id.sort_values('related_comments', ascending=[False]).reset_index()[['parent', 'related_comments']])
#print info
print(mostCommentedId)
commentid= mostCommentedId[['parent']].iloc[0]['parent']
print(commentid)
hacker_news.table_schema('full')

query = """SELECT type, count(ID) as number_of_type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_count=hacker_news.query_to_pandas_safe(query)
print(type_count.sort_values('number_of_type', ascending=[False]).reset_index()[['type', 'number_of_type']])
query = """SELECT COUNTIF(deleted) as deleted_comment
            FROM `bigquery-public-data.hacker_news.comments`
        """
deleted=hacker_news.query_to_pandas_safe(query)

print(deleted)

nb=deleted[['deleted_comment']].iloc[0]['deleted_comment']
print("Deleted comments: "+str(nb))

#different solution:
query = """SELECT deleted, COUNT(id) as number
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
deleted=hacker_news.query_to_pandas_safe(query)
print(deleted)