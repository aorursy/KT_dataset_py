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
from google.cloud import bigquery
client = bigquery.Client()

dataset_ref = client.dataset('stackoverflow', project='bigquery-public-data')

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)
table_posts_questions_ref = dataset_ref.table('posts_questions')

table_posts_questions = client.get_table(table_posts_questions_ref)

table_posts_questions.schema
query_1 = """

            SELECT COUNT(*) AS questions

            FROM `bigquery-public-data.stackoverflow.posts_questions` 

            WHERE answer_count > 0

            """

safe_query_1_job = client.query(query_1)

table_1 = safe_query_1_job.to_dataframe()

table_1
# ALL YEARS

count_questions_have_answers = table_1.questions[0]

count_questions = table_posts_questions.num_rows

print("count questions have answers = ", count_questions_have_answers, "; count all questions = ", count_questions)

print("the is percentage of questions that have been answered over the years ", count_questions_have_answers / count_questions * 100, "%")
# EVERY YEAR

query_1_1 = """

            SELECT EXTRACT(YEAR FROM creation_date) AS year,

                  ROUND(100 * SUM(IF(answer_count > 0, 1, 0)) / COUNT(*), 1) AS percentage_of_questions_with_answers

            FROM `bigquery-public-data.stackoverflow.posts_questions`

            GROUP BY year

            ORDER BY year

            """

safe_query_1_1_job = client.query(query_1_1)

table_1_1 = safe_query_1_1_job.to_dataframe()

table_1_1
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(14, 4))

sns.barplot(x='year', y='percentage_of_questions_with_answers', data=table_1_1)

table_users_ref = dataset_ref.table("users")

table_users = client.get_table(table_users_ref)

table_users.schema
# average reputation per year among all users

query_2 = """

            SELECT  EXTRACT(YEAR FROM creation_date) AS date,

                    AVG(reputation) AS reputation

            FROM `bigquery-public-data.stackoverflow.users`

            GROUP BY date

            ORDER BY date

            """

safe_query_2_job = client.query(query_2)

table_2 = safe_query_2_job.to_dataframe()

table_2
plt.figure(figsize=(14, 4))

sns.barplot(x='date', y='reputation', data=table_2)
table_badges_ref = dataset_ref.table('badges')

table_badges = client.get_table(table_badges_ref)

table_badges.schema
# number of badges per year for all users

query_3 = """

            SELECT  EXTRACT(YEAR FROM date) AS date,

                    COUNT(user_id) AS count_badges

            FROM `bigquery-public-data.stackoverflow.badges`

            GROUP BY date

            ORDER BY date

            """

safe_query_3_job = client.query(query_3)

table_3 = safe_query_3_job.to_dataframe()

table_3
plt.figure(figsize=(14, 4))

sns.barplot(x='date', y='count_badges', data=table_3)
query_4 = """

            SELECT  name AS name,

                    COUNT(user_id) AS gold

            FROM `bigquery-public-data.stackoverflow.badges`

            WHERE class = 1

            GROUP BY name

            ORDER BY gold DESC

            LIMIT 10

            """

safe_query_4_job = client.query(query_4)

table_4 = safe_query_4_job.to_dataframe()

table_4
plt.figure(figsize=(14, 4))

sns.barplot(x='name', y='gold', data=table_4)
query_5 = """

            SELECT  EXTRACT(DAYOFWEEK FROM q.creation_date) AS day_week,

                    SUM(IF(a.parent_id IS NOT NULL AND EXTRACT(MINUTE FROM q.creation_date) - EXTRACT(MINUTE FROM a.creation_date) < 60, 1, 0)) / COUNT(1) AS percent

            FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

            LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                 ON q.id = a.parent_id

            GROUP BY day_week

            ORDER BY percent

            """

safe_query_5_job = client.query(query_5)

table_5 = safe_query_5_job.to_dataframe()

table_5