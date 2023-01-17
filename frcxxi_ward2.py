import sys

sys.setrecursionlimit(10000)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

import sklearn

from scipy import stats

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch
import bq_helper

from bq_helper import BigQueryHelper

from google.cloud import bigquery

bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")

client = bigquery.Client()
QUERY = (

        f"""

            SELECT  

                c.n_comments /( DATE_DIFF('2020-03-01', DATE(u.creation_date), DAY)+1) as avg_comments_day,

                q.n_questions / (DATE_DIFF('2020-03-01', DATE(u.creation_date), DAY)+1) as avg_questions_day,

                a.n_answers / (DATE_DIFF('2020-03-01', DATE(u.creation_date), DAY)+1) as avg_answers_day

                

            FROM 

                bigquery-public-data.stackoverflow.users as u

                LEFT JOIN (

                    SELECT b.user_id, COUNT(*) as n_badges

                    FROM bigquery-public-data.stackoverflow.badges as b

                    GROUP BY b.user_id

                ) AS b ON b.user_id = u.id

                LEFT JOIN (

                    SELECT 

                        q.owner_user_id, 

                        COUNT(*) as n_questions, 

                        AVG(q.score) as avg_score_questions, 

                        AVG(q.answer_count) as avg_answers_questions, 

                        AVG(q.comment_count) as avg_comments_questions,

                        AVG(q.view_count) as avg_views_questions,

                        AVG(LENGTH(tags) - LENGTH(REPLACE(tags, '|', '')) + 1) as avg_tags_questions

                    FROM bigquery-public-data.stackoverflow.posts_questions as q

                    GROUP BY q.owner_user_id

                ) AS q ON q.owner_user_id = u.id

                LEFT JOIN (

                    SELECT 

                        a.owner_user_id, 

                        COUNT(*) as n_answers,

                        AVG(a.comment_count) as avg_comments_answers,

                        AVG(a.score) as avg_score_answers,

                    FROM bigquery-public-data.stackoverflow.posts_answers as a

                    GROUP BY a.owner_user_id

                ) AS a ON a.owner_user_id = u.id

                LEFT JOIN (

                    SELECT 

                        c.user_id, 

                        COUNT(*) as n_comments, 

                        AVG(c.score) as avg_score_comments

                    FROM bigquery-public-data.stackoverflow.comments as c

                    GROUP BY c.user_id

                ) AS c ON c.user_id = u.id

                

            WHERE u.creation_date <= '2020-03-01'

            

            ORDER BY RAND()

            LIMIT 5000

        """

    )

query_job = client.query(QUERY)  # API request

df = query_job.to_dataframe()  # Waits for query to finish

df

dataset_array = df.fillna(0).values

dataset_array
std_array = StandardScaler().fit_transform(dataset_array)

std_array
z = np.abs(stats.zscore(std_array))

std_array_clean = std_array[(z < 3).all(axis=1)]

len(std_array_clean)
z_values = pd.DataFrame(z)

index_good = z_values.applymap(lambda a: a<3).loc[lambda a: a.all(axis=1)].index

df_clean = df.loc[index_good]

df_clean
dendrogram = sch.dendrogram(sch.linkage(std_array_clean, method='ward', metric='euclidean'))
QUERY = (

        f"""

            SELECT 

                c.n_comments /( DATE_DIFF('2020-03-01', DATE(u.creation_date), DAY)+1) as avg_comments_day,

                q.n_questions / (DATE_DIFF('2020-03-01', DATE(u.creation_date), DAY)+1) as avg_questions_day,

                a.n_answers / (DATE_DIFF('2020-03-01', DATE(u.creation_date), DAY)+1) as avg_answers_day

                

            FROM 

                bigquery-public-data.stackoverflow.users as u

                LEFT JOIN (

                    SELECT b.user_id, COUNT(*) as n_badges

                    FROM bigquery-public-data.stackoverflow.badges as b

                    GROUP BY b.user_id

                ) AS b ON b.user_id = u.id

                LEFT JOIN (

                    SELECT 

                        q.owner_user_id, 

                        COUNT(*) as n_questions, 

                        AVG(q.score) as avg_score_questions, 

                        AVG(q.answer_count) as avg_answers_questions, 

                        AVG(q.comment_count) as avg_comments_questions,

                        AVG(q.view_count) as avg_views_questions,

                        AVG(LENGTH(tags) - LENGTH(REPLACE(tags, '|', '')) + 1) as avg_tags_questions

                    FROM bigquery-public-data.stackoverflow.posts_questions as q

                    GROUP BY q.owner_user_id

                ) AS q ON q.owner_user_id = u.id

                LEFT JOIN (

                    SELECT 

                        a.owner_user_id, 

                        COUNT(*) as n_answers,

                        AVG(a.comment_count) as avg_comments_answers,

                        AVG(a.score) as avg_score_answers,

                    FROM bigquery-public-data.stackoverflow.posts_answers as a

                    GROUP BY a.owner_user_id

                ) AS a ON a.owner_user_id = u.id

                LEFT JOIN (

                    SELECT 

                        c.user_id, 

                        COUNT(*) as n_comments, 

                        AVG(c.score) as avg_score_comments

                    FROM bigquery-public-data.stackoverflow.comments as c

                    GROUP BY c.user_id

                ) AS c ON c.user_id = u.id

                

            WHERE u.creation_date <= '2020-03-01'

            

            ORDER BY RAND()

            LIMIT 25000

        """

    )

query_job = client.query(QUERY)  # API request

df = query_job.to_dataframe()  # Waits for query to finish

df

dataset_array = df.fillna(0).values

dataset_array
std_array = StandardScaler().fit_transform(dataset_array)

std_array
z = np.abs(stats.zscore(std_array))

std_array_clean = std_array[(z < 3).all(axis=1)]

len(std_array_clean)
z_values = pd.DataFrame(z)

index_good = z_values.applymap(lambda a: a<3).loc[lambda a: a.all(axis=1)].index

df_clean = df.loc[index_good]

df_clean
km = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage ='ward')

res = km.fit_predict(std_array_clean)

(pd.Series(res).value_counts()/len(res))*100
df_clean["cluster"] = res



cluster = pd.DataFrame()



for n in set(res):

    cluster[n] = df_clean.loc[lambda a: a.cluster == n].mean()

    

cluster
std_cluster = pd.DataFrame()



for n in set(res):

    std_cluster[n] = pd.DataFrame(std_array_clean).loc[res == n].mean()

    

std_cluster.index = cluster.index[:3]



std_cluster



fig, ax = plt.subplots(figsize=(10,10))         

p2 = sns.heatmap(std_cluster, square=True, linewidths=.5, ax=ax, cmap='Blues_r')
std = pd.DataFrame(std_array_clean)

std



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = np.array(std[0])

y = np.array(std[1])

z = np.array(std[2])



ax.scatter(x,y,z, marker="s", c=df_clean["cluster"], s=40)



ax.set_xlabel('avg_questions_day')

ax.set_ylabel('avg_comments_day')

ax.set_zlabel('avg_answers_day')



plt.show()