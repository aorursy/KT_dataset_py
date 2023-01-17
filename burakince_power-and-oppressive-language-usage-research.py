from google.cloud import bigquery



client = bigquery.Client()

# If you query a BQ dataset that wasn't noboard, link you GCP account and setup the bigquery.Client() as follows:

# PROJECT_TO_RUN_JOBS = 'my-project-to-run-queries'

# client = bigquery.Client(project=PROJECT_TO_RUN_JOBS)



# List the tables in github_repos dataset which resides in bigquery-public-data project:

dataset = client.get_dataset('bigquery-public-data.github_repos')



tables = list(client.list_tables(dataset))

print([table.table_id for table in tables])
sql = '''

SELECT

  *

FROM

  `bigquery-public-data.github_repos.files`

LIMIT 5

'''



# Set up the query

query_job = client.query(sql)



# Make an API request  to run the query and return a pandas DataFrame

df = query_job.to_dataframe()

df.head()
import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import table



sql = '''

SELECT 'master or slave' as branch_name, count(distinct repo_name) as count

FROM `bigquery-public-data.github_repos.files`

WHERE ref IN ('refs/heads/master', 'refs/heads/slave')



UNION ALL



SELECT 'main' as branch_name, count(distinct repo_name) as count

FROM `bigquery-public-data.github_repos.files`

WHERE ref = 'refs/heads/main'

'''



query_job = client.query(sql)

df = query_job.to_dataframe()



plt.figure(figsize=(16,8))

# plot chart

ax1 = plt.subplot(121, aspect='equal')

df.plot(kind='pie', y = 'count', ax=ax1, autopct='%1.1f%%', startangle=90, shadow=False, labels=df['branch_name'], legend = False, fontsize=14)



# plot table

ax2 = plt.subplot(122)

plt.axis('off')

tbl = table(ax2, df, loc='center')

tbl.auto_set_font_size(True)

plt.show()
sql = '''

WITH oppressive_branch_named_repositories AS (

    SELECT repo_name

    FROM `bigquery-public-data.github_repos.files`

    WHERE ref IN ('refs/heads/master', 'refs/heads/slave')

    GROUP BY repo_name

),

new_repositories AS (

    SELECT repository_name

    FROM `bigquery-public-data.github_repos.commits`

    CROSS JOIN UNNEST(repo_name) as repository_name

    GROUP BY repository_name

    HAVING DATE(TIMESTAMP_SECONDS(MIN(committer.time_sec))) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH) AND CURRENT_DATE()

)



SELECT count(*) as count FROM new_repositories AS NEW_REPOS 

INNER JOIN oppressive_branch_named_repositories AS ALL_REPOS 

    ON NEW_REPOS.repository_name=ALL_REPOS.repo_name

'''



query_job = client.query(sql)



df = query_job.to_dataframe()

df.head()