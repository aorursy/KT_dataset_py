# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use("ggplot")

# Any results you write to the current directory are saved as output.
import bq_helper
github_repos = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                        dataset_name="github_repos")
# List all the tables in the database

github_repos.list_tables()
# lets check out the schema for the languages table

github_repos.table_schema("languages")
# head is used to get a snapshot for the first few rows of this table
github_repos.head("languages")
query1 = """SELECT repo_name, language FROM 
             `bigquery-public-data.github_repos.languages`
             WHERE repo_name LIKE "%tensorflow%";
             """
github_repos.estimate_query_size(query1)
lang_results = github_repos.query_to_pandas_safe(query1)
lang_results.head(10)
unique_programming_languages = []

for lang in lang_results['language']:
    for unique_lang in lang:
        unique_programming_languages.append(unique_lang["name"])
unique_programming_languages = pd.Series(unique_programming_languages)
unique_programming_languages.value_counts(normalize=True)[0:20].plot(kind="bar",figsize=(10,10),
                                                       title="Programming Language Count For Deep Learning Repositories"
                                                        , rot = 80)
github_repos.table_schema("sample_commits")
github_repos.head("sample_commits")
query2 = """SELECT subject, COUNT(*) AS commit_count
            FROM `bigquery-public-data.github_repos.sample_commits`
            WHERE repo_name = 'tensorflow/tensorflow'
            GROUP BY subject;"""
            
github_repos.estimate_query_size(query2)
q2 = github_repos.query_to_pandas_safe(query2)
q2.set_index("subject").sort_values("commit_count", ascending=True)[-10:].plot(kind="barh",
                                                                              title="Top 10 Subject Lines In Tensorflow Commit Messages")
github_repos.table_schema('commits')
query3 =  """SELECT author.name, COUNT(*) AS count FROM
            `bigquery-public-data.github_repos.commits`
            WHERE repo_name[OFFSET(0)] = 'tensorflow/tensorflow'
            GROUP BY author.name
            ; """
            
github_repos.estimate_query_size(query3)
q3 = github_repos.query_to_pandas(query3)
q3.set_index("name").sort_values('count', ascending= False)[0:5].plot(kind='bar')
q4 = """SELECT author.date, COUNT(*) AS count FROM
            `bigquery-public-data.github_repos.commits`
            WHERE repo_name[OFFSET(0)] = 'tensorflow/tensorflow'
            GROUP BY author.date
            ; """
github_repos.estimate_query_size(q4)
q4 = github_repos.query_to_pandas(q4)
q4.head()
q4.date = pd.to_datetime(q4.date)
q4.set_index('date').plot()
q5 = """SELECT message FROM
            `bigquery-public-data.github_repos.commits`
            WHERE repo_name[OFFSET(0)] = 'tensorflow/tensorflow'
            ; """
github_repos.estimate_query_size(q5)
q5 = github_repos.query_to_pandas(q5)
q5.head()
from wordcloud import WordCloud,STOPWORDS

text = q5.message.str.cat().lower()

stopwords = STOPWORDS

cloud = WordCloud(background_color="white",stopwords = stopwords,max_words=500,colormap='cubehelix').generate(text)

plt.figure(figsize = (8,7))
plt.imshow(cloud, interpolation='bilinear', aspect='auto')
plt.axis("off")
plt.title("Most Frequent Words In Tensorflow Commit Messages")
