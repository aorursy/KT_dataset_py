# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql.ex4 import *
print("Setup Complete")
from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "stackoverflow" dataset
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "posts_questions" table
table_ref = dataset_ref.table("posts_questions")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "posts_questions" table
client.list_rows(table, max_results=5).to_dataframe()
# Construct a reference to the "posts_answers" table
table_ref = dataset_ref.table("posts_answers")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "posts_answers" table
client.list_rows(table, max_results=5).to_dataframe()
# Number of users who posted both questions and answers
que_and_ans = """
select count( distinct q.owner_user_id)
from `bigquery-public-data.stackoverflow.posts_questions` q
inner join `bigquery-public-data.stackoverflow.posts_answers` a 
on q.owner_user_id = a.owner_user_id
"""
que_and_ans = client.query(que_and_ans).to_dataframe().iat[0,0]
print("Users that have posted questions as well as answers : {}".format(que_and_ans))
# Querying the number of users who only posted questions
only_que = """
select count( distinct q.owner_user_id)
from `bigquery-public-data.stackoverflow.posts_questions` q
left join `bigquery-public-data.stackoverflow.posts_answers` a 
on q.owner_user_id = a.owner_user_id
"""
only_que = client.query(only_que).to_dataframe().iat[0,0]
print("Users that have posted only questions but no answers : {}".format(only_que))
# Querying the number of users who only posted answers
only_ans = """
select count(distinct a.owner_user_id)
from `bigquery-public-data.stackoverflow.posts_answers` a
left join `bigquery-public-data.stackoverflow.posts_questions` q
on a.owner_user_id = q.owner_user_id
"""
only_ans = client.query(only_ans).to_dataframe().iat[0,0]
print("Users that have posted only answers and no questions : {}".format(only_ans))
# Total number of registered users on the stackoverflow platform(in the data)
users_number = """
select count(DISTINCT(u.id))
from `bigquery-public-data.stackoverflow.users` u
"""
users_num = client.query(users_number).to_dataframe().iat[0,0]
print("Total number of users registered on stackoverflow : {}".format(users_num))
print("% of users who post only questions : {:.2f}%".format((only_que/users_num)*100))
print("% of users who post only answers : {:.2f}%".format((only_ans/users_num)*100))
print("% of users who post both questions and answers : {:.2f}%".format((que_and_ans/users_num)*100))
# Query for extracting time period for which every question posted between '01-09-2013' and 
# '30-9-2013' was active
que_activity_time_period = """
select q.id AS q_id,
    min(timestamp_diff(q.last_activity_date,q.creation_date,DAY)) AS que_activity_time
from `bigquery-public-data.stackoverflow.posts_questions` q
where q.creation_date >= '2013-09-01' and q.creation_date <= '2013-09-30'
group by q_id
order by que_activity_time
"""
que_activity_time_period = client.query(que_activity_time_period).to_dataframe()
# Mean, Maximum and Minimum Question activity time(in days) for the month of September, 2013
que_activity_time_period['que_activity_time'].agg(['mean','max','min'])
# vote type, number of answers and comments on each stackoverflow posts
ans_comm_vote = """
select p.id AS p_id,p.title AS title,p.answer_count, p.comment_count, v.vote_type_id
from `bigquery-public-data.stackoverflow.stackoverflow_posts` p
    left join `bigquery-public-data.stackoverflow.votes` v
on p.id = v.id
where p.creation_date >= '2014-01-01' and p.creation_date <= '2014-01-31'
"""
ans_comm_vote = client.query(ans_comm_vote).to_dataframe()
ans_comm_vote.head()