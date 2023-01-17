# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
#hacker_news.head("comments")
#hacker_news.list_tables()
hacker_news.head("full")

# How many records are there of each type?
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

types = hacker_news.query_to_pandas_safe(query)
types.head()
# How many records of type comment have been deleted?
query2 = """
        SELECT id, COUNT(id)
        FROM `bigquery-public-data.hacker_news.full`
        WHERE type = 'comment' AND deleted IS TRUE
        GROUP BY id
        
        """

deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments.head()
deleted_comments.shape
# query3 = """
#         SELECT type, COUNT(id)
#         FROM `bigquery-public-data.hacker_news.full`
#         WHERE type = 'comment'
        
#         """